def test_check_synthetic_weather_dataset_import(setup_data):
    dataset = setup_data
    assert True


def test_check_environment_data(setup_data):
    sample = setup_data[0]
    invar, outvar = sample["invar"], sample["outvar"]
    assert invar.shape == outvar.shape
    assert [x for x in invar.shape] == [721 * 1440, 73]


def test_static_graph_data(setup_data):
    import torch

    _dataset = setup_data

    static_graph = _dataset.get_static_graph()

    # These values are obtained from the original paper. static_graph exposes
    # aggregated node/edge features (not raw src/dst index tensors); per-edge-type
    # connectivity is instead available via distributed_comm_patterns' local_edge_list,
    # which at world_size=1 contains every edge (no cross-rank partitioning).
    assert static_graph.mesh_level == 6
    assert static_graph.mesh_graph_node_features.shape == torch.Size([40962, 3])
    assert static_graph.mesh_graph_edge_features.shape == torch.Size([655320, 4])

    comm_patterns = static_graph.distributed_comm_patterns
    assert comm_patterns.mesh.local_edge_list.shape == torch.Size([655320, 2])
    assert comm_patterns.mesh.num_local_vertices == 40962

    assert static_graph.grid2mesh_graph_edge_features.shape == torch.Size([1618824, 4])
    assert comm_patterns.grid2mesh.local_edge_list.shape == torch.Size([1618824, 2])
    assert comm_patterns.grid2mesh.num_local_vertices == 40962

    assert static_graph.mesh2grid_graph_edge_features.shape == torch.Size([3114720, 4])
    assert comm_patterns.mesh2grid.local_edge_list.shape == torch.Size([3114720, 2])
    assert comm_patterns.mesh2grid.num_local_vertices == 721 * 1440
