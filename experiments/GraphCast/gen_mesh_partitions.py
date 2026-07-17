import os

import torch
from fire import Fire

from data_utils.graphcast_graph import DistributedGraphCastGraphGenerator
from data_utils.icosahedral_mesh import (
    get_hierarchy_of_triangular_meshes_for_sphere,
    merge_meshes,
)


def num_mesh_vertices(mesh_level: int) -> int:
    """Vertex count of the merged multi-mesh at this level.

    Mirrors DistributedGraphCastGraphGenerator.__init__'s own mesh construction
    rather than a closed-form formula, so it stays correct if mesh generation
    changes. Cross-checked against tests/test_single_graph_data.py: mesh_level=6
    -> 40962.
    """
    meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_level)
    merged = merge_meshes(meshes)
    return len(merged.vertices)


def _partition_path(output_dir: str, mesh_level: int, world_size: int) -> str:
    return os.path.join(
        output_dir, f"mesh_vertex_rank_placement_L{mesh_level}_W{world_size}.pt"
    )


def generate_partition(
    mesh_level: int, world_size: int, output_dir: str, force: bool = False
) -> str:
    """Compute (or reuse if cached) the mesh vertex rank placement for
    (mesh_level, world_size), saving it to output_dir. Returns the saved path."""
    os.makedirs(output_dir, exist_ok=True)
    path = _partition_path(output_dir, mesh_level, world_size)

    if os.path.exists(path) and not force:
        print(f"[skip] {path} already exists")
        return path

    if world_size == 1:
        placement = torch.zeros(num_mesh_vertices(mesh_level), dtype=torch.long)
    else:
        placement = DistributedGraphCastGraphGenerator.get_mesh_graph_partition(
            mesh_level=mesh_level, world_size=world_size
        )

    torch.save(placement, path)
    print(
        f"[saved] {path} (mesh_level={mesh_level}, world_size={world_size}, "
        f"num_vertices={placement.numel()})"
    )
    return path


def _parse_int_list(value) -> list:
    """Parse a CLI arg into a list of ints, tolerating Fire's auto-conversion.

    Fire turns "4,6" into a tuple (4, 6) and "1" into the int 1, so the value
    can arrive as a str, int, or tuple/list. Normalize all of these to
    list[int] so `--mesh_levels="4,6"`, `--mesh_levels=4`, and
    `--mesh_levels 4 6` all work.
    """
    if isinstance(value, str):
        return [int(x) for x in value.split(",") if x != ""]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def main(
    mesh_levels: str = "2,5,6",
    world_sizes: str = "1,2,4",
    output_dir: str = "partitions",
    force: bool = False,
) -> None:
    """Generate/cache mesh partitions for the cross product of mesh_levels x world_sizes.

    Args:
        mesh_levels: comma-separated list of mesh levels, e.g. "2,5,6".
        world_sizes: comma-separated list of world sizes, e.g. "1,2,4".
        output_dir: directory to write mesh_vertex_rank_placement_L{L}_W{W}.pt files to.
        force: recompute and overwrite even if a cached file already exists.
    """
    levels = _parse_int_list(mesh_levels)
    sizes = _parse_int_list(world_sizes)

    for mesh_level in levels:
        for world_size in sizes:
            generate_partition(mesh_level, world_size, output_dir, force=force)


if __name__ == "__main__":
    # Usage: python gen_mesh_partitions.py --mesh_levels 2,5,6 --world_sizes 1,2,4 --output_dir partitions/
    Fire(main)
