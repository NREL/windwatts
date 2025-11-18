from typing import List

def generate_key(lat: float, lon: float, height: List[int], yearly: bool) -> str:
    """
    Generate a unique key for the database based on the parameters.

    Args:
        lat (float): Latitude of the location
        lon (float): Longitude of the location
        height (List[int]): List of heights in integer
        yearly (bool): Boolean flag to indicate to return yearly averaged data or latest row of data

    Returns:
        str: A unique key for the database.
    """
    height_str = '_'.join(map(str, height))
    return f"{lat}_{lon}_{height_str}_{'yearly' if yearly else 'latest'}"

def format_coordinate(coordinate: float) -> str:
    """Format a coordinate to 3 decimal places, matching JS .toFixed(3)."""
    return f"{coordinate:.3f}"

def chunker(file_obj, chunk_size: int = 1024 * 1024):
    """Stream the spooled file in chunks for constant memory usage"""
    while True:
        data = file_obj.read(chunk_size)
        if not data:
            break
        yield data