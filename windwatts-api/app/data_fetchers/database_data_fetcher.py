import json
from .abstract_data_fetcher import WTKDataFetcher
from app.utils.data_fetcher_utils import generate_key


class DatabaseDataFetcher(WTKDataFetcher):
    """
    Class for fetching data from the database.
    TODO: Refactor this to handle pre-computed/aggregated results in the future instead of caching.
    """

    def __init__(self, db_manager):
        """
        Initializes the DatabaseDataFetcher with the given DatabaseManager.

        Args:
            db_manager (DatabaseManager): The DatabaseManager to use for fetching data.
        """
        self.db_manager = db_manager

    def fetch_data(self, lat: float, lng: float, height: int):
        """
        Fetch data from the database.

        Args:
            lat (float): Latitude of the location
            lng (float): Longitude of the location
            height (int): Heights in meters

        Returns:
            dict: The fetched data
        """
        key = generate_key(lat, lng, height)
        data = self.db_manager.get_data(key)
        return json.loads(data) if data else []

    def store_data(self, lat: float, lng: float, height: int, data: str):
        """
        Store data in the database.

        Args:
            lat (float): Latitude of the location
            lng (float): Longitude of the location
            height (int): Heights in meters
            data (str): The data to be stored
        """
        key = generate_key(lat, lng, height)
        self.db_manager.store_data(key, data)
