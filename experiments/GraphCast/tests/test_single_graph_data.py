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

    # These values are obtained from the original paper
    assert static_graph.mesh_level == 6
    assert static_graph.mesh_graph_node_features.shape == torch.Size([40962, 3])
    assert static_graph.mesh_graph_edge_features.shape == torch.Size([655320, 4])
    assert static_graph.mesh_graph_src_indices.shape == torch.Size([655320])
    assert static_graph.mesh_graph_dst_indices.shape == torch.Size([655320])

    assert static_graph.grid2mesh_graph_edge_features.shape == torch.Size([1618822, 4])
    assert static_graph.grid2mesh_graph_src_indices.shape == torch.Size([1618822])
    assert static_graph.grid2mesh_graph_dst_indices.shape == torch.Size([1618822])

    assert static_graph.grid2mesh_graph_edge_features.shape == torch.Size([1618822, 4])
    assert static_graph.grid2mesh_graph_src_indices.shape == torch.Size([1618822])
    assert static_graph.grid2mesh_graph_dst_indices.shape == torch.Size([1618822])
