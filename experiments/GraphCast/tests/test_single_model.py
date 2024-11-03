def test_full_model(setup_data):
    import sys
    import os
    import torch

    cur_dir = os.path.dirname(__file__)
    prev_dir = os.path.dirname(cur_dir)
    sys.path.append(prev_dir)
    from model import DGraphCast
    import graphcast_config
    from dist_utils import SingleProcessDummyCommunicator

    cfg = graphcast_config.Config()
    comm = SingleProcessDummyCommunicator()
    # Create the dataset
    dataset = setup_data
    batch_size = 1
    # Create the model
    model = DGraphCast(cfg, comm)
