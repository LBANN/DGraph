import pytest


@pytest.fixture(scope="module")
def setup_data():
    import sys
    import os

    cur_dir = os.path.dirname(__file__)
    prev_dir = os.path.dirname(cur_dir)
    sys.path.append(prev_dir)
    from dataset import SyntheticWeatherDataset

    latlon_res = (721, 1440)
    num_samples_per_year_train = 2
    num_workers = 8
    num_channels_climate = 73
    num_history = 0
    dt = 6.0
    start_year = 1980
    use_time_of_year_index = True
    channels_list = [i for i in range(num_channels_climate)]

    cos_zenith_args = {
        "dt": dt,
        "start_year": start_year,
    }
    batch_size = 1
    test_dataset = SyntheticWeatherDataset(
        channels=channels_list,
        num_samples_per_year=num_samples_per_year_train,
        num_steps=1,
        grid_size=latlon_res,
        cos_zenith_args=cos_zenith_args,
        batch_size=batch_size,
        num_workers=num_workers,
        num_history=num_history,
        use_time_of_year_index=use_time_of_year_index,
    )
    return test_dataset


def test_check_synthetic_weather_dataset_import(setup_data):
    dataset = setup_data
    assert True


def test_check_environment_data(setup_data):
    sample = setup_data[0]
    invar, outvar = sample["invar"], sample["outvar"]
    assert invar.shape == outvar.shape
    assert [x for x in invar.shape] == [73, 721, 1440]
