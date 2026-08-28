import pytest


@pytest.fixture(scope="session")
def setup_data():
    import sys
    import os
    import torch

    cur_dir = os.path.dirname(__file__)
    prev_dir = os.path.dirname(cur_dir)
    sys.path.append(prev_dir)
    from dataset import SyntheticWeatherDataset
    from data_utils.icosahedral_mesh import (
        get_hierarchy_of_triangular_meshes_for_sphere,
        merge_meshes,
    )

    latlon_res = (721, 1440)
    num_samples_per_year_train = 2
    num_workers = 8
    num_channels_climate = 73
    num_history = 0
    dt = 6.0
    start_year = 1980
    use_time_of_year_index = True
    channels_list = [i for i in range(num_channels_climate)]
    mesh_level = 6

    cos_zenith_args = {
        "dt": dt,
        "start_year": start_year,
    }
    batch_size = 1

    # world_size=1 placement: every mesh vertex is owned by rank 0. Computed by
    # mirroring DistributedGraphCastGraphGenerator's own mesh construction rather
    # than a hardcoded vertex count, so it stays correct if mesh_level changes.
    _meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_level)
    _merged_mesh = merge_meshes(_meshes)
    mesh_vertex_placement = torch.zeros(len(_merged_mesh.vertices), dtype=torch.long)

    test_dataset = SyntheticWeatherDataset(
        channels=channels_list,
        num_samples_per_year=num_samples_per_year_train,
        num_steps=1,
        mesh_vertex_placement=mesh_vertex_placement,
        mesh_level=mesh_level,
        grid_size=latlon_res,
        cos_zenith_args=cos_zenith_args,
        batch_size=batch_size,
        num_workers=num_workers,
        num_history=num_history,
        use_time_of_year_index=use_time_of_year_index,
        device=torch.device("cpu"),
    )
    return test_dataset
