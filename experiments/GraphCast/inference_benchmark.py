import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from fire import Fire

_GRAPHCAST_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_GRAPHCAST_DIR)
_COST_MODEL_DIR = os.path.join(_EXPERIMENTS_DIR, "cost_model_benchmarks")
sys.path.insert(0, _COST_MODEL_DIR)

from benchmarks.common import (  # noqa: E402
    collect_metadata,
    setup_distributed,
    write_result,
)

from DGraph.Communicator import Communicator  # noqa: E402
from DGraph.utils.TimingReport import TimingReport  # noqa: E402

from data_utils.graphcast_graph import (  # noqa: E402
    DistributedGraphCastGraphGenerator,
    move_graphcast_graph_to_device,
)
from data_utils.utils import padded_size  # noqa: E402
from dataset import SyntheticWeatherDataset  # noqa: E402
from dist_utils import SingleProcessDummyCommunicator  # noqa: E402
from graphcast_config import Config  # noqa: E402
from model import DGraphCast  # noqa: E402

GRID_LAT, GRID_LON = 721, 1440
NUM_GRID_POINTS = GRID_LAT * GRID_LON


def build_comm(mode: str, backend: str, world_size: int):
    """mode="single" -> SingleProcessDummyCommunicator (no real halo exchange).
    mode="distributed" -> real DGraph Communicator (NCCL). Both branches assume
    torch.distributed is already initialized (see setup_distributed() in main())."""
    if mode == "single":
        return SingleProcessDummyCommunicator()
    return Communicator.init_process_group(backend, ranks_per_graph=world_size)


def load_partition(partition_dir: str, mesh_level: int, world_size: int) -> torch.Tensor:
    path = os.path.join(
        partition_dir, f"mesh_vertex_rank_placement_L{mesh_level}_W{world_size}.pt"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing partition file {path}. Generate it first with:\n"
            f"  python gen_mesh_partitions.py --mesh_levels {mesh_level} "
            f"--world_sizes {world_size} --output_dir {partition_dir}"
        )
    return torch.load(path)


def build_dataset(
    cfg: Config,
    mesh_level: int,
    partition: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
) -> SyntheticWeatherDataset:
    return SyntheticWeatherDataset(
        channels=list(range(cfg.model.input_grid_dim)),
        num_samples_per_year=2,
        num_steps=1,
        mesh_vertex_placement=partition,
        mesh_level=mesh_level,
        grid_size=(GRID_LAT, GRID_LON),
        rank=rank,
        world_size=world_size,
        ranks_per_graph=world_size,
        device=device,
    )


def summarize(trials_ms) -> dict:
    if not trials_ms:
        return {"mean": None, "std": None, "p50": None, "p95": None, "trials_ms": []}
    arr = np.asarray(trials_ms, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "trials_ms": [float(x) for x in trials_ms],
    }


def collect_phase_breakdown(measure_iters: int, processor_layers: int) -> dict:
    """Reads DGraph.utils.TimingReport.TimingReport._timers, already populated by
    model.py's existing TimingReport(...) blocks during run_forward_loop -- no new
    instrumentation needed. Model control flow is deterministic (no data-dependent
    branching), so the last `measure_iters` entries per name correspond exactly to
    the measured (post-warmup) iterations."""
    names = [
        "model/embed",
        "model/encode",
        "model/process",
        "model/decode",
        "model/final_prediction",
        "encoder/halo_exchange",
        "encoder/edge_block",
        "encoder/node_block",
        "encoder/grid_mlp",
        "decoder/halo_exchange",
        "decoder/edge_block",
        "decoder/node_block",
    ]
    for i in range(processor_layers):
        names += [
            f"processor/layer_{i}/halo_exchange",
            f"processor/layer_{i}/edge_block",
            f"processor/layer_{i}/node_block",
        ]

    breakdown = {}
    for name in names:
        values = TimingReport._timers.get(name, [])
        values = values[-measure_iters:] if measure_iters > 0 else values
        breakdown[name] = summarize(values)
    return breakdown


def run_forward_loop(
    model: torch.nn.Module,
    in_data: torch.Tensor,
    static_graph,
    comm,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
) -> dict:
    model.eval()
    e2e_trials_ms = []
    oom = False

    try:
        for i in range(warmup_iters + measure_iters):
            if i == warmup_iters:
                torch.cuda.reset_peak_memory_stats(device)

            comm.barrier()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            with torch.no_grad():
                pred = model(in_data, static_graph)
            end_evt.record()
            torch.cuda.synchronize()
            comm.barrier()

            if i >= warmup_iters:
                e2e_trials_ms.append(start_evt.elapsed_time(end_evt))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom = True
        else:
            raise

    peak_memory_bytes = None if oom else torch.cuda.max_memory_allocated(device)
    return {
        "e2e_trials_ms": e2e_trials_ms,
        "peak_memory_bytes": peak_memory_bytes,
        "oom": oom,
    }


def run_correctness_check(
    cfg: Config,
    mesh_level: int,
    comm,
    partition_dir: str,
    rank: int,
    world_size: int,
    device: torch.device,
    seed: int,
    atol: float,
    rtol: float,
) -> bool:
    """Two-step correctness gate for world_size > 1 runs.

    Step 1 (structural): SyntheticWeatherDataset.__getitem__ shards grid input/
    output via a naive contiguous chunk, independent of
    DistributedGraphCastGraphGenerator's connectivity-weighted grid_part voting
    used to build the mesh2grid CommunicationPattern. If their per-rank local
    grid vertex counts disagree, GraphCastDecoder silently operates on the
    wrong slice of grid vertices. This step catches that mismatch and halts
    before the (more expensive, more failure-prone) Step 2.

    Step 2 (value-level, only if Step 1 passes): every rank independently
    builds an identical, full (world_size=1, SingleProcessDummyCommunicator)
    reference forward pass -- redundant across ranks, but necessary because
    DGraph.distributed.commInfo.compute_comm_map's dist.all_gather is a
    collective on the REAL process group and must be called identically by
    every rank to avoid deadlock. The reference and distributed runs use
    different internal vertex renumberings (each DistributedGraphCastGraphGenerator
    call re-derives its own grid renumbering for its own world_size), so both
    are mapped back to a common "original grid index" frame before comparing.
    """
    partition = load_partition(partition_dir, mesh_level, world_size)

    num_local_expected = padded_size(NUM_GRID_POINTS, world_size) // world_size

    dataset = build_dataset(cfg, mesh_level, partition, rank, world_size, device)
    static_graph = dataset.get_static_graph()
    num_local_actual = static_graph.distributed_comm_patterns.mesh2grid.num_local_vertices

    gathered = [None] * world_size
    dist.all_gather_object(gathered, (rank, num_local_expected, num_local_actual))
    step1_ok = all(e == a for (_, e, a) in gathered)

    if rank == 0:
        if step1_ok:
            print("[correctness] STEP 1 PASS: dataset chunk size vs grid_part local "
                  "vertex count agree on every rank.")
        else:
            mismatches = [(r, e, a) for (r, e, a) in gathered if e != a]
            print(f"[correctness] STEP 1 FAIL: per-rank (rank, expected, actual) "
                  f"mismatches: {mismatches}")
            print("[correctness] Halting -- do not trust world_size>1 benchmark "
                  "numbers until this is resolved (see plan's known dataset.py vs "
                  "grid_part mismatch).")
    dist.barrier()
    if not step1_ok:
        return False

    # --- Step 2: value-level check ---
    lat_lon_grid = dataset.lat_lon_grid

    # Recover this run's grid renumbering (original grid index for each
    # renumbered/rank-sorted slot) without re-triggering comm-pattern construction.
    dist_generator = DistributedGraphCastGraphGenerator(
        lat_lon_grid, mesh_level=mesh_level, ranks_per_graph=world_size,
        rank=rank, world_size=world_size,
    )
    dist_mesh_graph = dist_generator.get_mesh_graph(partition)
    dist_grid2mesh = dist_generator.get_grid2mesh_graph(dist_mesh_graph)
    dist_renumbered_grid = dist_grid2mesh["renumbered_grid"]  # [N_grid] -> original grid idx
    dist_grid_placement = dist_grid2mesh["grid_vertex_rank_placement"]  # sorted by rank

    local_mask = dist_grid_placement == rank
    local_original_grid_idx = dist_renumbered_grid[local_mask]

    # Every rank independently builds an identical world_size=1 reference (all
    # ranks call the same dist collectives the same number of times -> no deadlock).
    torch.manual_seed(seed)
    ref_comm = SingleProcessDummyCommunicator()
    ref_generator = DistributedGraphCastGraphGenerator(
        lat_lon_grid, mesh_level=mesh_level, ranks_per_graph=1, rank=0, world_size=1
    )
    ref_placement = torch.zeros(ref_generator.mesh_vertices.shape[0], dtype=torch.long)
    ref_static_graph = ref_generator.get_graphcast_graph(
        mesh_vertex_rank_placement=ref_placement
    )
    # Built on CPU; move onto the compute device to match ref_model / ref_input.
    ref_static_graph = move_graphcast_graph_to_device(ref_static_graph, device)
    ref_mesh_graph = ref_generator.get_mesh_graph(ref_placement)
    ref_grid2mesh = ref_generator.get_grid2mesh_graph(ref_mesh_graph)
    ref_renumbered_grid = ref_grid2mesh["renumbered_grid"]

    torch.manual_seed(seed)
    ref_model = DGraphCast(cfg, ref_comm).to(device).eval()

    canonical_input = torch.randn(NUM_GRID_POINTS, cfg.model.input_grid_dim)
    ref_input = canonical_input[ref_renumbered_grid].unsqueeze(0).to(device)

    with torch.no_grad():
        ref_pred = ref_model(ref_input, ref_static_graph)  # [N_grid, C], ref-renumbered order

    ref_pred_original_order = torch.zeros_like(ref_pred)
    ref_pred_original_order[ref_renumbered_grid.to(device)] = ref_pred

    expected_local = ref_pred_original_order[local_original_grid_idx.to(device)]

    torch.manual_seed(seed)
    dist_model = DGraphCast(cfg, comm).to(device).eval()
    dist_input = canonical_input[local_original_grid_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        actual_local = dist_model(dist_input, static_graph)

    close = torch.allclose(actual_local, expected_local, atol=atol, rtol=rtol)
    max_abs_diff = (actual_local - expected_local).abs().max().item()
    denom = expected_local.abs().clamp_min(1e-8)
    max_rel_diff = ((actual_local - expected_local).abs() / denom).max().item()

    gathered2 = [None] * world_size
    dist.all_gather_object(gathered2, (rank, bool(close), max_abs_diff, max_rel_diff))
    dist.barrier()

    if rank == 0:
        all_close = all(c for (_, c, _, _) in gathered2)
        if all_close:
            print(f"[correctness] STEP 2 PASS: all ranks match reference within "
                  f"atol={atol}, rtol={rtol}. Per-rank (rank, close, max_abs_diff, "
                  f"max_rel_diff): {gathered2}")
        else:
            print(f"[correctness] STEP 2 FAIL. Per-rank (rank, close, max_abs_diff, "
                  f"max_rel_diff): {gathered2}")
        return all_close
    return close


def main(
    mode: str = "distributed",
    mesh_level: int = 6,
    backend: str = "nccl",
    batch_size: int = 1,
    warmup_iters: int = 10,
    measure_iters: int = 50,
    partition_dir: str = "partitions",
    output_dir: str = "benchmark_results",
    seed: int = 42,
    correctness: bool = False,
    correctness_atol: float = 1e-4,
    correctness_rtol: float = 1e-3,
) -> None:
    assert mode in ("single", "distributed"), "mode must be 'single' or 'distributed'"
    assert batch_size == 1, "only batch_size=1 is supported (matches GraphCast's one-step inference)"
    assert not correctness or mode == "distributed", (
        "--correctness checks cross-rank partitioning; run it with --mode distributed "
        "(and --nproc_per_node >= 2), not --mode single"
    )

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    comm = build_comm(mode, backend, world_size)

    cfg = Config()
    cfg.model.mesh_level = mesh_level

    # model.py's DGraphCast.forward unconditionally uses TimingReport(...) context
    # managers, which raise if TimingReport.init() hasn't been called -- required
    # before ANY forward pass, in both the correctness-check and benchmark paths.
    TimingReport.init(comm)

    if correctness:
        if world_size < 2:
            if rank == 0:
                print("[correctness] world_size < 2: nothing to check (single-GPU "
                      "baseline has no cross-rank partitioning to verify).")
            return
        ok = run_correctness_check(
            cfg, mesh_level, comm, partition_dir, rank, world_size, device,
            seed, correctness_atol, correctness_rtol,
        )
        if rank == 0:
            print(f"[correctness] Overall: {'PASS' if ok else 'FAIL'}")
        return

    partition = load_partition(partition_dir, mesh_level, world_size)
    dataset = build_dataset(cfg, mesh_level, partition, rank, world_size, device)
    static_graph = dataset.get_static_graph()

    torch.manual_seed(seed)
    model = DGraphCast(cfg, comm).to(device).eval()

    sample = dataset[0]
    in_data = sample["invar"].unsqueeze(0).to(device)

    result = run_forward_loop(
        model, in_data, static_graph, comm, warmup_iters, measure_iters, device
    )

    # Use a collective to agree on OOM across ranks: if only one rank OOMs (e.g. a
    # slightly imbalanced METIS partition), letting ranks diverge onto different
    # code paths here would deadlock the next collective call (all_gather_object
    # below) on whichever ranks didn't OOM and are still waiting for it.
    oom_flags = [None] * world_size
    dist.all_gather_object(oom_flags, result["oom"])
    any_oom = any(oom_flags)

    if any_oom:
        if rank == 0:
            print(f"[inference_benchmark] OOM at mode={mode} mesh_level={mesh_level} "
                  f"world_size={world_size} (per-rank oom={oom_flags}) -- infeasible "
                  f"at this scale on this hardware, not a benchmark failure.")
            payload = {
                "benchmark": "graphcast_inference",
                "metadata": collect_metadata(),
                "config": {
                    "mode": mode, "backend": backend, "world_size": world_size,
                    "mesh_level": mesh_level, "hidden_dim": cfg.model.hidden_dim,
                    "processor_layers": cfg.model.processor_layers,
                    "batch_size": batch_size, "warmup_iters": warmup_iters,
                    "measure_iters": measure_iters, "seed": seed,
                },
                "measurements": [{
                    "params": {"world_size": world_size, "mesh_level": mesh_level},
                    "oom": True,
                    "oom_per_rank": oom_flags,
                }],
            }
            write_result(
                os.path.join(output_dir, f"graphcast_infer_{mode}_W{world_size}_L{mesh_level}.json"),
                payload,
            )
        return

    latency_summary = summarize(result["e2e_trials_ms"])
    phase_breakdown = collect_phase_breakdown(measure_iters, cfg.model.processor_layers)

    peak_memory_list = [None] * world_size
    dist.all_gather_object(peak_memory_list, result["peak_memory_bytes"])

    trials_list = [None] * world_size
    dist.all_gather_object(trials_list, result["e2e_trials_ms"])

    if rank == 0:
        p50_s = (latency_summary["p50"] or 0.0) / 1000.0
        throughput = (NUM_GRID_POINTS * batch_size / p50_s) if p50_s > 0 else None

        payload = {
            "benchmark": "graphcast_inference",
            "metadata": collect_metadata(),
            "config": {
                "mode": mode, "backend": backend, "world_size": world_size,
                "mesh_level": mesh_level, "hidden_dim": cfg.model.hidden_dim,
                "processor_layers": cfg.model.processor_layers,
                "batch_size": batch_size, "warmup_iters": warmup_iters,
                "measure_iters": measure_iters, "seed": seed,
            },
            "measurements": [{
                "params": {"world_size": world_size, "mesh_level": mesh_level},
                "end_to_end_latency_ms": latency_summary,
                "throughput_grid_points_per_sec": throughput,
                "peak_memory_bytes": {
                    "per_rank": peak_memory_list,
                    "max": max(m for m in peak_memory_list if m is not None),
                },
                "phase_breakdown_ms": phase_breakdown,
                "per_rank_end_to_end_trials_ms": trials_list,
            }],
        }
        write_result(
            os.path.join(output_dir, f"graphcast_infer_{mode}_W{world_size}_L{mesh_level}.json"),
            payload,
        )
        print(
            f"[inference_benchmark] mode={mode} world_size={world_size} "
            f"mesh_level={mesh_level}: p50={latency_summary['p50']:.2f}ms "
            f"throughput={throughput:.1f} grid-points/sec"
        )


if __name__ == "__main__":
    Fire(main)
