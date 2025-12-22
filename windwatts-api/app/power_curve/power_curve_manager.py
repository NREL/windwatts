from .powercurve import PowerCurve
import os
import pandas as pd
import calendar
import numpy as np
from scipy.interpolate import CubicSpline
from enum import Enum
from typing import Optional

class DatasetSchema(Enum):
    TIMESERIES = "timeseries"                # Any raw time-series data (year/month/hour based, magnitudes not quantiles) - wtk
    QUANTILES_WITH_YEAR = "quantiles_with_year"  # Quantile distributions, separated by year - era5
    QUANTILES_GLOBAL = "quantiles_global"        # Quantile distribution without year (global) - ensemble data

class PowerCurveManager:
    """
    Manages multiple power curves stored in a directory.
    """
    def __init__(self, 
                power_curve_dir: str,
                use_swi_default: bool = False,
                schema_swi_prefs: Optional[dict] = None):
        """
        Initialize PowerCurveManager to load multiple power curves.

        :param power_curve_dir: Directory containing power curve files.
        :param use_swi_default: Fallback if a schema isn't in schema_swi_prefs.
        :param schema_swi_prefs: Optional dict overriding per-schema SWI behavior.       
        """
        self.power_curves = {}
        self.load_power_curves(power_curve_dir)
        self.use_swi_default = use_swi_default

        self.schema_swi_prefs = {
            DatasetSchema.TIMESERIES: False, # not used for midpoints; defined for completeness
            DatasetSchema.QUANTILES_WITH_YEAR: True, # SWI ON
            DatasetSchema.QUANTILES_GLOBAL: False # SWI OFF
        }
        
        if schema_swi_prefs:
            self.schema_swi_prefs.update(schema_swi_prefs)
        
    def _use_swi_for(self, schema: DatasetSchema) -> bool:
        """Resolve SWI strictly by schema (falls back to class default if absent)."""
        return self.schema_swi_prefs.get(schema, self.use_swi_default)
    
    def set_schema_swi_pref(self, schema: DatasetSchema, enabled: bool) -> None:
        """Change SWI default for a specific schema."""
        self.schema_swi_prefs[schema] = bool(enabled)

    # ---------- NEW: schema detection ----------
    def _classify_schema(self, df: pd.DataFrame) -> DatasetSchema:
        """
        Classify the dataset schema based on column patterns.

        This inspects the DataFrame's column names (case-insensitive) to determine
        what kind of dataset structure it represents. The classification is used
        later to decide which average types (global, yearly, monthly, hourly) are
        supported.

        Detection logic:
        - If the dataset has a ``probability`` column → it's a quantile dataset.
            - If it also has a ``year`` column → QUANTILES_WITH_YEAR
            (separate distributions per year).
            - Otherwise → QUANTILES_GLOBAL
            (single global quantile distribution, no time(year, month or hour) separation).
        - If there is no ``probability`` column → assume WTK-style time-series data.
            - If it has a combined ``mohr`` column (month+hour encoding), or
            separate ``month`` and ``hour`` columns → TIMESERIES.
        - Fallback: if none of the expected markers are found, default to
        TIMESERIES, but subsequent processing may still raise errors if critical
        columns are missing.

        :param df: Input DataFrame with schema to classify.
        :type df: pd.DataFrame
        :return: A DatasetSchema enum value indicating the schema type.
        :rtype: DatasetSchema
        """
        cols = set(df.columns.str.lower())  # robust to case
        has_prob = "probability" in cols
        has_year = "year" in cols
        has_mohr = "mohr" in cols
        has_month_hour = "month" in cols and "hour" in cols

        if has_prob:
            if has_year:
                return DatasetSchema.QUANTILES_WITH_YEAR
            else:
                return DatasetSchema.QUANTILES_GLOBAL
        # No probability column → treat as WTK-like time series
        if has_mohr or has_month_hour:
            return DatasetSchema.TIMESERIES
        # Fall back: if neither, assume WTK-like (time-series) but raise if critical cols are missing later
        return DatasetSchema.TIMESERIES
    
    def load_power_curves(self, directory: str):
        """
        Load power curves from the specified directory.
        """
        for file in os.listdir(directory):
            if file.endswith(".csv") or file.endswith(".xlsx"):
                curve_name = os.path.splitext(file)[0]
                self.power_curves[curve_name] = PowerCurve(os.path.join(directory, file))

    def get_curve(self, curve_name: str) -> PowerCurve:
        """
        Retrieves a power curve by name.

        Args:
            curve_name (str): Name of the power curve.

        Returns:
            PowerCurve: Corresponding power curve object.
        """
        if curve_name not in self.power_curves:
            raise KeyError(f"Power curve '{curve_name}' not found.")
        return self.power_curves[curve_name]
    
    def find_inverse(self, x_smooth: np.ndarray, y_smooth: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        """
        Vectorized inverse mapping: finds the x values corresponding to the closest y values to each y_hat.

        :param x_smooth: Smoothed x values.
        :type x_smooth: numpy.ndarray
        :param y_smooth: Smoothed y values corresponding to x_smooth.
        :type y_smooth: numpy.ndarray
        :param y_hat: Target y values to invert.
        :type y_hat: numpy.ndarray

        :return: Array of x values corresponding to closest y matches.
        :rtype: numpy.ndarray
        """
        # Broadcasting to compute pairwise absolute differences
        diff = np.abs(y_smooth[:, None] - y_hat[None, :])
        closest_indices = np.argmin(diff, axis=0)
        return x_smooth[closest_indices]
    
    def _jitter_nonincreasing(self, q: np.ndarray, eps: float = 1e-5):
        """
        Ensure q is strictly increasing by adding a tiny epsilon to any element
        that is <= its predecessor. Long flat runs become a tiny staircase.
        
        :param q: Input array of quantile values that may include equal or decreasing entries.
        :type q: numpy.ndarray
        :param eps: Minimum increment applied to enforce strict monotonicity. Defaults to 1e-5.
        :type eps: float
        :return: A strictly increasing version of the input array with minimal perturbation.
        :rtype: numpy.ndarray
        """
        # Example:
        # q = np.array([3.02, 3.02, 3.02, 3.05])
        # _jitter_nonincreasing(q, eps=1e-5)
        # array([3.02, 3.02001, 3.02002, 3.05])
        q = np.asarray(q, dtype=np.float64).copy()
        for i in range(1, q.size):
            if not np.isfinite(q[i]) or not np.isfinite(q[i-1]):
                continue
            if q[i] <= q[i-1]:
                q[i] = q[i-1] + eps
        return q
    
    def run_cubic(self, x, y, probs_new, M1):
        """
        Internal helper: fit cubic spline F(q)=P(X≤q) and invert to Q(p).

        Given strictly increasing quantile values (`x`) and their corresponding
        probabilities (`y`), this fits a cubic spline with clamped endpoint slopes,
        samples it on a dense grid, ensures the resulting CDF is monotone, and
        numerically inverts it to return a smooth quantile function Q(p).

        :param x: Quantile values (must be strictly increasing).
        :type x: numpy.ndarray
        :param y: Cumulative probabilities corresponding to x.
        :type y: numpy.ndarray
        :param probs_new: Uniformly spaced probabilities at which to estimate new quantiles.
        :type probs_new: numpy.ndarray
        :param M1: Number of interpolation points for CDF smoothing.
        :type M1: int
        :return: Tuple (quantiles_new, probs_new) representing the smoothed quantile curve.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        # === Compute approximate derivatives at endpoints ===
        dy_start = (y[1] - y[0]) / (x[1] - x[0])  # Forward difference
        dy_end = (y[-1] - y[-2]) / (x[-1] - x[-2])  # Backward difference

        # === Create the cubic spline with clamped boundary conditions ===
        spline = CubicSpline(x, y, bc_type=((1, dy_start), (1, dy_end)))
        
        #=== High-resolution discretization (interp_point_count is large) ===
        x_smooth = np.linspace(x[0], x[-1], M1, dtype=np.float64)
        y_smooth = spline(x_smooth)

        # Invert F(q) -> Q(p)
        q_new = self.find_inverse(x_smooth, y_smooth, probs_new)

        return q_new,probs_new
    
    def estimation_quantiles_SWI(self, quantiles, probs, M1=1000, M2=501):
        """
        Estimate a smoother quantile function using the Spline With Inversion (SWI) method, with a safe fallback..

        This method constructs a cubic spline interpolation of the empirical CDF (defined by the 
        provided `quantiles` and corresponding `probs`), and then performs an inversion to generate 
        a smooth estimate of quantiles over a high-resolution, uniformly spaced probability range.

        If the cubic spline fails due to equal or non-increasing quantile values
        (which violate the strictly increasing requirement of CubicSpline),
        those quantiles are adjusted by a small epsilon (+1e-5) to enforce
        monotonicity before retrying. This correction is minimal and does not
        materially affect the resulting averages.

        Assumes that:
            - `quantiles` and `probs` are both sorted in ascending order.
            - `probs` span the interval [0, 1] and are uniformly spaced.

        :param quantiles: Observed quantile values (e.g., from sample data).
        :type quantiles: numpy.ndarray
        :param probs: Corresponding cumulative probabilities for the quantiles.
        :type probs: numpy.ndarray
        :param M1: Number of points for spline interpolation (default: 1000).
        :type M1: int
        :param M2: Number of evenly spaced probability points at which to estimate new quantiles (default: 501).
        :type M2: int

        :return: Tuple containing:
            - quantiles_new (numpy.ndarray): Estimated quantile values corresponding to probs_new.
            - probs_new (numpy.ndarray): Uniformly spaced probabilities in [0, 1] (length M2).
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """

        q = np.asarray(quantiles, dtype=np.float64)  # quantiles
        p = np.asarray(probs,     dtype=np.float64)  # probabilities

        # Predefine defaults in case both attempts fail
        probs_new = np.linspace(0, 1, M2, dtype=np.float64)
        quantiles_new_default = np.zeros_like(probs_new, dtype=np.float64)
        
        try:
            return self.run_cubic(q, p, probs_new, M1)
        except (ValueError, ZeroDivisionError) as e1:
            print(f"Cubic Spline failed due to: {e1}. Attempting Fallback...")
        except Exception as e2:
            print(f"Cubic Spline failed due to: {e2}.")
            return quantiles_new_default, probs_new
        
        try:
            q_fix = self._jitter_nonincreasing(q, eps=1e-5)
            return self.run_cubic(q_fix, p, probs_new, M1)
        except Exception as e3:
            print(f"Warning: CubicSpline failed even after jittering — {e3}")
            return quantiles_new_default, probs_new
    
    def _quantiles_to_kw_midpoints(
        self,
        df_sorted: pd.DataFrame,
        ws_col: str,
        power_curve: PowerCurve,
        use_swi: bool
    ) -> pd.DataFrame:
        """
        Takes a dataframe with columns [probability, ws_col] sorted by probability
        → smooth CDF via SWI or not based on use_swi flag → midpoint quantiles → kW via power curve.
        Returns a dataframe with columns [ws_col, f"{ws_col}_kw"] for equal-probability midpoints.
        """
        probs = df_sorted["probability"].to_numpy(dtype=float)
        quants = df_sorted[ws_col].to_numpy(dtype=float)

        if use_swi:
            q_est, _ = self.estimation_quantiles_SWI(quantiles=quants,probs=probs)
        else:
            q_est = quants

        qs = pd.Series(q_est, dtype=float)
        midpoints = (qs.shift(-1) + qs) / 2
        midpoints = midpoints.iloc[:-1]  # drop last NaN

        # Convert to kW
        mid_df = pd.DataFrame({ws_col: midpoints})
        mid_df[f"{ws_col}_kw"] = power_curve.windspeed_to_kw(mid_df, ws_col)
        return mid_df

    def compute_energy_production_df(self, df: pd.DataFrame, height: int, selected_power_curve: str, relevant_columns_only: bool = True) -> pd.DataFrame:
        """
        Computes energy production dataframe using the selected power curve.

        Args:
            df (pd.DataFrame): Dataframe containing wind speed data.
            height (int): Height in meters for which to estimate power production.
            selected_power_curve (str): Name of the selected power curve.
            relevant_columns_only (bool): If True, returns only relevant columns.

        Returns:
            pd.DataFrame
            - WTK-like: ["year","month","hour", ws_col, f"{ws_col}_kw"] (if relevant_columns_only)
            - Quantiles-with-year: ["year", ws_col, f"{ws_col}_kw"] for midpoint bins
            - Global-quantiles: ["year"(absent), ws_col, f"{ws_col}_kw"] for midpoint bins
        """
        ws_col = f'windspeed_{height}m'
        if ws_col not in df.columns:
            raise KeyError(f"Expected column '{ws_col}' in input dataframe.")
        
        schema = self._classify_schema(df)
        power_curve = self.get_curve(selected_power_curve)

        if schema == DatasetSchema.TIMESERIES:
            # era5 data doesn't have month and hour columns
            work = df.copy()
            if "month" not in work.columns or "hour" not in work.columns:
                if "mohr" not in work.columns:
                    raise KeyError("WTK-like input requires 'mohr' or explicit 'month' and 'hour' columns.")
                work["month"], work["hour"] = work["mohr"] // 100, work["mohr"] % 100

            work[f"{ws_col}_kw"] = power_curve.windspeed_to_kw(work, ws_col)
            if relevant_columns_only:
                cols = ["year", "mohr", "month", "hour", ws_col, f"{ws_col}_kw"]
                return work[cols]
            return work
        
        elif schema == DatasetSchema.QUANTILES_WITH_YEAR:
            use_swi_eff = self._use_swi_for(schema)
            records = []
            for year, group in df.groupby("year"):
                # sorting by probability is important since the records might be shuffled by "groupby" and we are using midpoint method.
                group = group.sort_values("probability").reset_index(drop=True)
                mid_df = self._quantiles_to_kw_midpoints(group, ws_col, power_curve, use_swi=use_swi_eff)
                mid_df.insert(0, "year", year)
                records.append(mid_df)
            out = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["year", ws_col, f"{ws_col}_kw"])
            return out if not relevant_columns_only else out[["year", ws_col, f"{ws_col}_kw"]]
        
        else:  # DatasetSchema.QUANTILES_GLOBAL
            use_swi_eff = self._use_swi_for(schema)
            group = df.sort_values("probability").reset_index(drop=True)
            out = self._quantiles_to_kw_midpoints(group, ws_col, power_curve, use_swi=use_swi_eff)
            return out if not relevant_columns_only else out[[ws_col, f"{ws_col}_kw"]]
    
    def prepare_yearly_production_df(self, df: pd.DataFrame, height: int, selected_power_curve: str) -> pd.DataFrame:
        """
        Prepares yearly average energy production and windspeed dataframe for dependent methods.

        Args:
            df (pd.DataFrame): Dataframe containing data at all heights for a location.
            height (int): Height in meters.
            selected_power_curve (str): Power curve

        Returns:
            Returns a dataframe with ["year","Average wind speed (m/s)","kWh produced"].
            For global quantiles (no year), returns a single pseudo-row with year=None.
            pd.Dataframe
        """
        prod_df = self.compute_energy_production_df(df, height, selected_power_curve)
        ws_column = f'windspeed_{height}m'
        kw_column = f'windspeed_{height}m_kw'

        schema = self._classify_schema(df)
        
        res_list = []
        if schema == DatasetSchema.TIMESERIES:
            work = prod_df.copy()
            # If wind direction columns slipped through, drop them
            work = work.drop(columns=[c for c in work.columns if "winddirection" in c], errors="ignore")

            for year, group in work.groupby("year"):
                avg_ws = group[ws_column].mean()
                # Original approximation used in your code:
                # sum of instantaneous power over typical month × 30 days
                kwh = group[kw_column].sum() * 30
                res_list.append({
                    "year": year,
                    "Average wind speed (m/s)": avg_ws,
                    "kWh produced": kwh
                })

        elif schema == DatasetSchema.QUANTILES_WITH_YEAR:
            # Midpoints are equal-probability bins → average power × hours/year
            for year, group in prod_df.groupby("year"):
                avg_ws = group[ws_column].mean()
                avg_power_kw = group[kw_column].mean()
                kwh = avg_power_kw * 8760.0
                res_list.append({
                    "year": year,
                    "Average wind speed (m/s)": avg_ws,
                    "kWh produced": kwh
                })

        else:  # QUANTILES_GLOBAL
            if len(prod_df) == 0:
                return pd.DataFrame(columns=["year", "Average wind speed (m/s)", "kWh produced"])

            avg_ws = prod_df[ws_column].mean()
            avg_power_kw = prod_df[kw_column].mean()
            kwh = avg_power_kw * 8760.0
            res_list.append({
                "year": None,
                "Average wind speed (m/s)": avg_ws,
                "kWh produced": kwh
            })

        res = pd.DataFrame(res_list)
        res.sort_values("Average wind speed (m/s)", inplace=True, ignore_index=True)
        return res
    
    def calculate_yearly_energy_production(self, df: pd.DataFrame, height: int, selected_power_curve: str) -> dict:
        """
        Computes yearly average energy production and windspeed.

        Args:
            df (pd.DataFrame): Dataframe containing data at all heights for a location.
            height (int): Height in meters.
            selected_power_curve (str): Power curve

        Returns:
            dict
        
        Example:
            {
                "2001": {"Average wind speed (m/s)": "5.65", "kWh produced": 250117},
                "2002": {"Average wind speed (m/s)": "5.72", "kWh produced": 264044},
                ...
            }
        """
        yearly_prod_df = self.prepare_yearly_production_df(df,height,selected_power_curve)
        
        result = {}
        for _, row in yearly_prod_df.iterrows():
            # Use "Global" if year is missing (for quantiles without year)
            year_key = "Global" if pd.isna(row["year"]) else str(int(row["year"]))
            result[year_key] = {
                "Average wind speed (m/s)": f"{float(row['Average wind speed (m/s)']):.2f}",
                "kWh produced": int(round(float(row["kWh produced"])))
            }

        return result
    
    def calculate_energy_production_summary(self, df: pd.DataFrame, height: int, selected_power_curve: str) -> dict:
        """
        Computes yearly average energy production and windspeed summary.

        Args:
            df (pd.DataFrame): Dataframe containing data at all heights for a location.
            height (int): Height in meters.
            selected_power_curve (str): Power curve

        Returns:
            dict
        
        Example:
            {
                "Lowest year": {"year": 2015, "Average wind speed (m/s)": "5.36", "kWh produced": 202791},
                "Average year": {"year": None, "Average wind speed (m/s)": "5.86", "kWh produced": 267712},
                "Highest year": {"year": 2014, "Average wind speed (m/s)": "6.32", "kWh produced": 326354}
            }
        """
        yearly_prod_df = self.prepare_yearly_production_df(df,height,selected_power_curve)
        if yearly_prod_df.empty:
            return {}
        res_avg = pd.DataFrame(yearly_prod_df.drop(columns=['year']).mean()).T
        res_avg.index = ["Average year"]

        # Final formatting
        res_summary = pd.concat([yearly_prod_df.iloc[[0]], res_avg, yearly_prod_df.iloc[[-1]]], ignore_index=False)

        def fmt_year(v):
            return None if pd.isna(v) else int(v)

        # Handle None year for average row - convert to proper None instead of pandas NA
        res_summary["year"] = res_summary["year"].map(fmt_year)
        res_summary["kWh produced"] = res_summary["kWh produced"].astype(float).round().astype(int)
        res_summary["Average wind speed (m/s)"] = res_summary["Average wind speed (m/s)"].astype(float).map('{:,.2f}'.format)

        res_summary.index = [
            "Lowest year",
            "Average year",
            "Highest year"
        ]
        res_summary = res_summary.replace({np.nan: None})
        return res_summary.to_dict(orient="index")

    def calculate_monthly_energy_production(self, df: pd.DataFrame, height: int, selected_power_curve: str) -> dict:
        """
        Computes monthly average energy production.

        Args:
            df (pd.DataFrame): Dataframe containing data at all heights for a location.
            height (int): Height in meters.
            selected_power_curve (str): Power curve
            data_type (str): data source {wtk or era5}
        Returns:
            dict: dict summarizing monthly energy production and windspeed.

        Example:
        {'Jan': {'Average wind speed, m/s': '3.80', 'kWh produced': '5,934'}, 
        'Feb': {'Average wind speed, m/s': '3.92', 'kWh produced': '6,357'}, 
        'Mar': {'Average wind speed, m/s': '4.17', 'kWh produced': '7,689'}....}
        """
        schema = self._classify_schema(df)
        if schema != DatasetSchema.TIMESERIES:
            raise ValueError("Monthly averages are only supported for time-series (TIMESERIES) inputs.")
        prod_df = self.compute_energy_production_df(df, height, selected_power_curve)
       
        ws_column = f'windspeed_{height}m'
        kw_column = f'windspeed_{height}m_kw'

        work = prod_df.drop(columns=["mohr", "year", "hour"] + [col for col in prod_df.columns if "winddirection" in col],errors="ignore")
        
        res = work.groupby("month").agg(avg_ws=(ws_column, "mean"), kwh_total=(kw_column, "sum"))
        res["kwh_total"] *= 30 / 20.0  # Approximation: 30 days per month, averaged over 20 years

        res.rename(columns={"avg_ws": "Average wind speed (m/s)", "kwh_total": "kWh produced"}, inplace=True)
        res.index = pd.Series(res.index).apply(lambda x: calendar.month_abbr[int(x)])

        res["kWh produced"] = res["kWh produced"].round().astype(int)
        res["Average wind speed (m/s)"] = res["Average wind speed (m/s)"].astype(float).map('{:,.2f}'.format)

        return res.to_dict(orient="index")