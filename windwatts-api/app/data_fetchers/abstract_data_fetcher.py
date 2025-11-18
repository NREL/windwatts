from abc import ABC, abstractmethod
from typing import List

class AbstractDataFetcher(ABC):
    """
    Abstract class for fetching data from the WTK API
    """
    def __init__(self):
        """
        Add default settings, configurations, etc.
        """
        pass

    @abstractmethod
    def fetch_data(self, lat: float, lng: float, height: int):
        """
        Data fetching method specifications:
        
        Args:
            lat (float): Latitude of the location
            lng (float): Longitude of the location
            height (int): Height in integer

        Returns:
            dict: A dictionary containing the fetched data
        """
        pass