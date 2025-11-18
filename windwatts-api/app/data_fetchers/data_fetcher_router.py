import json
from .abstract_data_fetcher import AbstractDataFetcher

class DataFetcherRouter:
    def __init__(self):
        """
        Initializes the DataFetcherRouter.
        """
        self.fetchers = {}

    def register_fetcher(self, fetcher_name: str, fetcher: AbstractDataFetcher):
        """
        Register a data fetcher with the router.

        Args:
            fetcher_name (str): The name of the fetcher.
            fetcher: The fetcher object.
        """
        self.fetchers[fetcher_name] = fetcher

    def fetch_data(self, params: dict , source: str = "athena_wtk"):
        """
        Fetch data using specified data fetcher.

        Args:
            params (dict): The parameters to pass to the fetcher, including
                lat (float): Latitude of the location
                lng (float): Longitude of the location
                height (int): Heights in meters
            source (str): The name of the fetcher to use

        Returns:
            dict: The fetched data as a dictionary.
        """
        fetcher = self.fetchers.get(source)
        if fetcher:
            return fetcher.fetch_data(**params)
        else:
            raise ValueError(f"No fetcher found for source={source}")

    def fetch_data_routing(self, params: dict, source: str = "athena_wtk"):
        """
        Fetch data using the appropriate data fetcher through routing logics.

        Args:
            params (dict): The parameters to pass to the fetcher.
            source (str): The name of the fetcher to use
        Returns:
            dict: The fetched data as a dictionary.
        """
        data = None
        cached_data = None

        # Check available fetchers by name and type
        db_key = "database"
        s3_key = "s3"
        athena_key = source

        db_fetcher_available = db_key in self.fetchers
        s3_fetcher_available = s3_key in self.fetchers
        athena_fetcher_available = athena_key in self.fetchers

        if db_fetcher_available:
            cached_data = self.fetchers[db_key].fetch_data(**params)
            if cached_data:
                return cached_data

        if athena_fetcher_available and self.is_complex_query(params):
            data = self.fetchers[athena_key].fetch_data(**params)
        elif s3_fetcher_available:
            data = self.fetchers[s3_key].fetch_data(**params)

        if db_fetcher_available and not cached_data and data:
            self.fetchers[db_key].store_data(**params, data=json.dumps(data))

        return data or {}

    @staticmethod
    def is_complex_query(params: dict) -> bool:
        """
        Determine if the query is complex.

        Args:
            lat (float): Latitude of the location.
            lng (float): Longitude of the location.
            height (List[int]): List of heights in meters.
            yearly (bool): Whether to fetch yearly averaged data.

        Returns:
            bool: True if the query is complex, False otherwise.
        """
        # Implement logic to determine if the query is complex
        # For example, if the query spans multiple files or requires aggregation
        return True