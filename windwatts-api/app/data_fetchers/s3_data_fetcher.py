from typing import List, Optional
import boto3
import pandas as pd
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config
from .abstract_data_fetcher import AbstractDataFetcher

MIN_POOL_WORKERS = 1

s3_key_templates = {
    "era5" : "{prefix}/year={year}/index={index}/{year}_{index}.csv.gz",
    "wtk" : "{prefix}/year={year}/varset=all/index={index}/{index}_{year}_all.csv.gz"
}

class S3DataFetcher(AbstractDataFetcher):
    def __init__(self, bucket_name: str, prefix: str, s3_key_template: str, grid: str):
        """
        Initializes the S3DataFetcher with the given bucket.
        Parameters:
            bucket_name (str): The s3 bucket name.
            prefix (str): The s3 prefix the specifies folder inside a bucket.
            s3_key_template(str): The s3 key template to download files.
            grid (str): The grid of the data. "era5" or "wtk"
        """
        print(f"Initializing S3 Data Fetcher: bucket: {bucket_name} prefix: {prefix} grid: {grid} ...")
        self.s3_client = boto3.client(
            "s3",
            config=Config(
                retries={"max_attempts": 5, "mode": "standard"},
                tcp_keepalive=True,
            ),
        )
        self.bucket = bucket_name
        self.prefix = prefix
        self.grid = grid
        if s3_key_template not in s3_key_templates:
            raise ValueError(f"Unknown s3_key_template '{s3_key_template}', expected one of {list(s3_key_templates)}")
        self.s3_key_template = s3_key_templates[s3_key_template]
    
    def generate_s3_keys(self, grid_Indices: List[str], years: List[int]) -> List[str]:
        """
        Build S3 object keys for the given years and nearest grid indices.

        :param gridIndices: list of grid indices of neighbors.
        :param years: data of years to fetch.
        :return: List of S3 object keys.
        """
        keys: List[str] = []

        for year in years:
            for index in grid_Indices:
                # uri specific to era5 timeseries, might change for ensemble timeseries as it might not have year.
                key = self.s3_key_template.format(prefix=self.prefix, year=year, index=index)
                keys.append(key)
        
        return keys
    
    def fetch_s3_file(self, key: str, cols: Optional[List[str]]):
        """
        Download + parse a single gzip CSV from S3.

        :param key: S3 object key.
        :param cols: Optional list of column names to load (passed to pandas.read_csv).
        :return: DataFrame or None on error.
        """
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            payload = obj["Body"].read()
            df = pd.read_csv(
                BytesIO(payload),
                compression="gzip",
                usecols=cols,
            )
            return df
        except Exception as e:
            print(f"Warning: Failed to fetch {key}: {e}")
            return None

    def fetch_data(
        self,
        gridIndices: List[str],
        years: List[int],
        cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch data from S3 in parallel (optimized for ~5 files).

        Args:
            gridIndexes (List(str)): Grid indices of neighbors with respect user selected coordinate.
            years (List[int]): Years of which the data is needed.
            cols (Optional[List[str]]): Optional column subset to read.

        Returns:
            pd.DataFrame: Concatenated DataFrame across requested files (may be empty).
        """
        keys = self.generate_s3_keys(gridIndices, years)
        if not keys:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        workers = min(MIN_POOL_WORKERS, len(keys))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self.fetch_s3_file, k, cols) for k in keys]
            for f in as_completed(futures):
                df = f.result()
                if df is not None and not df.empty:
                    frames.append(df)
        if not frames:
            print(f"No data found for gridIndices {gridIndices}, years={years}")
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)

        return out
