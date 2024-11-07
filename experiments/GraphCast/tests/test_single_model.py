def test_full_model(setup_data):
    """Tests a single pass through the full model."""
    import sys
    import os
    import torch

    cur_dir = os.path.dirname(__file__)
    prev_dir = os.path.dirname(cur_dir)
    sys.path.append(prev_dir)
    from model import DGraphCast
    import graphcast_config
    from dist_utils import SingleProcessDummyCommunicator, CommAwareDistributedSampler
    from torch.utils.data import DataLoader

    cfg = graphcast_config.Config()
    comm = SingleProcessDummyCommunicator()
    # Create the dataset
    dataset = setup_data
    batch_size = 1
    # Create the model
    model = DGraphCast(cfg, comm)
    sampler = CommAwareDistributedSampler(dataset, comm)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    _data = next(iter(dataloader))
    in_data, out_data = _data["invar"], _data["outvar"]
    static_graph = dataset.get_static_graph()

    model.train()
    pred = model(in_data, static_graph)

    assert pred.unsqueeze(0).shape == out_data.shape
