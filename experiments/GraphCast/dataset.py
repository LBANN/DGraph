import torch
import numpy as np
import time
from typing import Any, Dict, List, Tuple
from torch.utils.data import Dataset


class SyntheticWeatherDataset(Dataset):
    """
    A dataset for generating synthetic temperature data on a latitude-longitude grid for multiple atmospheric layers.

    Args:
        channels (list): List of channels representing different atmospheric layers.
        num_samples_per_year (int): Total number of days to simulate per year.
        num_steps (int): Number of consecutive days in each training sample.
        grid_size (tuple): Latitude by longitude dimensions of the temperature grid.
        base_temp (float): Base temperature around which variations are simulated.
        amplitude (float): Amplitude of the sinusoidal temperature variation.
        noise_level (float): Standard deviation of the noise added to temperature data.
        **kwargs: Additional keyword arguments for advanced configurations.
    """

    def __init__(
        self,
        channels: List[int],
        num_samples_per_year: int,
        num_steps: int,
        device: str | torch.device = "cuda",
        grid_size: Tuple[int, int] = (721, 1440),
        base_temp: float = 15,
        amplitude: float = 10,
        noise_level: float = 2,
        **kwargs: Any,
    ):
        self.num_days: int = num_samples_per_year
        self.num_steps: int = num_steps
        self.num_channels: int = len(channels)
        self.device = device
        self.grid_size: Tuple[int, int] = grid_size
        start_time = time.time()
        self.temperatures: np.ndarray = self.generate_data(
            self.num_days,
            self.num_channels,
            self.grid_size,
            base_temp,
            amplitude,
            noise_level,
        )
        print(
            f"Generated synthetic temperature data in {time.time() - start_time:.2f} seconds."
        )
        self.extra_args: Dict[str, Any] = kwargs

    def generate_data(
        self,
        num_days: int,
        num_channels: int,
        grid_size: Tuple[int, int],
        base_temp: float,
        amplitude: float,
        noise_level: float,
    ) -> np.ndarray:
        """
        Generates synthetic temperature data over a specified number of days for multiple atmospheric layers.

        Args:
            num_days (int): Number of days to generate data for.
            num_channels (int): Number of channels representing different layers.
            grid_size (tuple): Grid size (latitude, longitude).
            base_temp (float): Base mean temperature for the data.
            amplitude (float): Amplitude of temperature variations.
            noise_level (float): Noise level to add stochasticity to the temperature.

        Returns:
            numpy.ndarray: A 4D array of temperature values across days, channels, latitudes, and longitudes.
        """
        days = np.arange(num_days)
        latitudes, longitudes = grid_size

        # Create altitude effect and reshape
        altitude_effect = np.arange(num_channels) * -0.5
        altitude_effect = altitude_effect[
            :, np.newaxis, np.newaxis
        ]  # Shape: (num_channels, 1, 1)
        altitude_effect = np.tile(
            altitude_effect, (1, latitudes, longitudes)
        )  # Shape: (num_channels, latitudes, longitudes)
        altitude_effect = altitude_effect[
            np.newaxis, :, :, :
        ]  # Shape: (1, num_channels, latitudes, longitudes)
        altitude_effect = np.tile(
            altitude_effect, (num_days, 1, 1, 1)
        )  # Shape: (num_days, num_channels, latitudes, longitudes)

        # Create latitude variation and reshape
        lat_variation = np.linspace(-amplitude, amplitude, latitudes)
        lat_variation = lat_variation[:, np.newaxis]  # Shape: (latitudes, 1)
        lat_variation = np.tile(
            lat_variation, (1, longitudes)
        )  # Shape: (latitudes, longitudes)
        lat_variation = lat_variation[
            np.newaxis, np.newaxis, :, :
        ]  # Shape: (1, 1, latitudes, longitudes)
        lat_variation = np.tile(
            lat_variation, (num_days, num_channels, 1, 1)
        )  # Shape: (num_days, num_channels, latitudes, longitudes)

        # Create time effect and reshape
        time_effect = np.sin(2 * np.pi * days / 365)
        time_effect = time_effect[
            :, np.newaxis, np.newaxis, np.newaxis
        ]  # Shape: (num_days, 1, 1, 1)
        time_effect = np.tile(
            time_effect, (1, num_channels, latitudes, longitudes)
        )  # Shape: (num_days, num_channels, latitudes, longitudes)

        # Generate noise
        noise = np.random.normal(
            scale=noise_level, size=(num_days, num_channels, latitudes, longitudes)
        )

        # Calculate daily temperatures
        daily_temps = base_temp + altitude_effect + lat_variation + time_effect + noise

        return daily_temps

    def __len__(self) -> int:
        """
        Returns the number of samples available in the dataset.
        """
        return self.num_days - self.num_steps

    def __getitem__(self, idx: int):
        """
        Retrieves a sample from the dataset at the specified index.
        """
        return [
            {
                "invar": torch.tensor(self.temperatures[idx], dtype=torch.float32).to(
                    self.device
                ),
                "outvar": torch.tensor(
                    self.temperatures[idx + 1 : idx + self.num_steps + 1],
                    dtype=torch.float32,
                ).to(self.device),
            }
        ]


def test_synthetic_weather_dataset(num_days):
    latlon_res = (721, 1440)
    num_samples_per_year_train = num_days
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
    test_dataset = SyntheticWeatherDataset(
        channels=channels_list,
        num_samples_per_year=num_samples_per_year_train,
        num_steps=1,
        grid_size=latlon_res,
        cos_zenith_args=cos_zenith_args,
        batch_size=1,
        num_workers=num_workers,
        num_history=num_history,
        use_time_of_year_index=use_time_of_year_index,
    )
    print(len(test_dataset))


if __name__ == "__main__":
    from fire import Fire

    Fire(test_synthetic_weather_dataset)
