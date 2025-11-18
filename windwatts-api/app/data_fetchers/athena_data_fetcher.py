from .abstract_data_fetcher import AbstractDataFetcher
from windwatts_data import WindwattsWTKClient, WindwattsERA5Client, WindwattsEnsembleClient

class AthenaDataFetcher(AbstractDataFetcher):
    def __init__(self, athena_config: str, source_key: str):
        """
        Initializes the AthenaDataFetcher with a single source_key like 'wtk', 'era5', or 'era5_bc'.
        We infer the base family ('wtk' or 'era5') from the part before the first underscore.

        Args:
            athena_config (str): Path to the Athena configuration file.
            source_key (str): Key in the config that specifies which athena source.
        """
        # self.data_type = data_type.lower()
        self.source_key = source_key.lower()
        self.base_type = self.source_key.split("_", 1)[0]  # 'wtk' or 'era5' (from 'era5_bc' too)

        if self.base_type == 'wtk':
            print(f"Initializing WTK Client with Source Key: {self.source_key}")
            self.client = WindwattsWTKClient(config_path=athena_config, source_key = self.source_key) # source_key  "wtk"
        elif self.base_type == 'era5':
            print(f"Initializing ERA5 Client with Source Key: {self.source_key}")
            self.client = WindwattsERA5Client(config_path=athena_config, source_key = self.source_key) # source_key  "era5" or "era5_bc"
        elif self.base_type == 'ensemble':
            print(f"Initializing Ensemble Client with Source Key: {self.source_key}")
            self.client = WindwattsEnsembleClient(config_path=athena_config, source_key = self.source_key) # source_key  "ensemble"
        else:
            raise ValueError(f"Unsupported base dataset: {self.base_type}")

    def fetch_data(self, lat: float, lng: float, height: int, avg_type: str = 'global') -> dict:
        """
        Fetch wind data using the configured client.

        Args:
            lat (float): Latitude of the location.
            lng (float): Longitude of the location.
            height (int): Height in meters.
            avg_type (str): Average type to fetch.
                For 'wtk': ['global', 'yearly', 'monthly', 'hourly', 'none']
                For 'era5': ['global', 'yearly', 'none']

        Returns:
            dict: Fetched wind data.

        Raises:
            ValueError: If the avg_type is not supported for the selected client.
        """
        # valid_avg_types_params = {
        #     'wtk': ['global', 'yearly', 'monthly', 'hourly', 'none'],
        #     'era5': ['global', 'yearly', 'none'],
        #     'era5_bc': ['global','none']
        # }

        # if avg_type not in valid_avg_types_params[self.source_key]:
        #     raise ValueError(f"avg_type '{avg_type}' is not supported for data type '{self.data_type}'")

        if avg_type == 'none':
            return self.client.fetch_df(lat=lat, long=lng, height=height)
        elif avg_type == 'global':
            return self.client.fetch_global_avg_at_height(lat=lat, long=lng, height=height)
        elif avg_type == 'yearly':
            return self.client.fetch_yearly_avg_at_height(lat=lat, long=lng, height=height)
        elif avg_type == 'monthly':
            return self.client.fetch_monthly_avg_at_height(lat=lat, long=lng, height=height)
        elif avg_type == 'hourly':
            return self.client.fetch_hourly_avg_at_height(lat=lat, long=lng, height=height)
        else:
            raise ValueError(f"Invalid avg_type: {avg_type}")
    
    def find_nearest_locations(self, lat: float, lng: float, n_neighbors: int = 1):
        """
        Find one or more nearest grid locations (index, latitude, and longitude) to a given coordinate.

        :param lat: Latitude of the target location in decimal degrees.
        :type lat: float
        :param lng: Longitude of the target location in decimal degrees.
        :type lng: float
        :param n_neighbors: Number of nearest grid points to return. Defaults to 1.
        :type n_neighbors: int

        :return: 
            - If n_neighbors == 1: a list of single tuple [(index, latitude, longitude)] for the nearest grid point.  
            - If n_neighbors > 1: a list of tuples, each containing (index, latitude, longitude).
            - The list will have length n_neighbors.
        :rtype: 
            :rtype: list[tuple[str, float, float]]
        """
        # A list of tuples where each tuple contains: (grid_index, latitude, longitude)
        tuples = self.client.find_n_nearest_locations(lat, lng, n_neighbors)
        return tuples