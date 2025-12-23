import pandas as pd
import numpy as np
from scipy import interpolate


class PowerCurve(object):
    def __init__(self, power_curve_path):
        # Load data and minimal preprocessing

        if ".xslx" in power_curve_path:
            self.raw_data = pd.read_excel(power_curve_path)
            self.raw_data.rename(
                columns={"Wind Speed (m/s)": "ws", "Turbine Output": "kw"}, inplace=True
            )
        elif ".csv" in power_curve_path:
            self.raw_data = pd.read_csv(power_curve_path)
            self.raw_data.rename(
                columns={"Wind Speed (m/s)": "ws", "Turbine Output": "kw"}, inplace=True
            )
            # print(self.raw_data.columns)
        else:
            raise ValueError(
                "Unsupported powercurve file format (should be .xslx or .csv)."
            )

        # Add (0,0) if not there already
        if self.raw_data["ws"].min() > 0:
            self.raw_data.loc[len(self.raw_data)] = [0, 0]
            self.raw_data = self.raw_data.sort_values("ws", ascending=True)
            self.raw_data.reset_index(drop=True, inplace=True)

        # Create vectors for interpolation
        self.interp_x = self.raw_data.ws
        self.interp_y = self.raw_data.kw

        # Cubic interpolation
        # self.powercurve_intrp = interp1d(self.interp_x, self.interp_y, kind='cubic')
        # Switched back to linear to avoid bad interpolation with negative values
        self.powercurve_intrp = interpolate.interp1d(
            self.interp_x, self.interp_y, kind="linear"
        )

        # Saving a list of instances where windspeeds are higher/lower than what is in the curve
        self.above_curve = []
        self.below_curve = []

        self.max_ws = max(self.raw_data.ws)

        self.reset_counters()

    def windspeed_to_kw(
        self, df, ws_column="ws-adjusted", dt_column="datetime", trim=True
    ):
        """Converts wind speed to kw"""

        # by default round down/up values below or under the range of the curve
        if trim:
            ws = (
                df[ws_column]
                .apply(lambda x: 0 if x < 0 else x)
                .apply(lambda x: self.max_ws if x > self.max_ws else x)
            )
        else:
            ws = df[ws_column]

        kw = self.powercurve_intrp(ws)

        below_curve = df[kw < 0]
        above_curve = df[kw > self.max_ws]

        if dt_column in df.columns:
            self.below_curve.extend(
                zip(below_curve[dt_column].tolist(), below_curve[ws_column].tolist())
            )
            self.above_curve.extend(
                zip(above_curve[dt_column].tolist(), above_curve[ws_column].tolist())
            )

        return kw

    def reset_counters(self):
        self.above_curve = []
        self.below_curve = []

    # def plot(self):
    #     fig = px.line(
    #         y=self.powercurve_intrp(self.interp_x),
    #         x=self.interp_x,
    #         labels={"x": "Windspeed (m/s)", "y": "Power (kW)"},
    #     )
    #     fig.add_trace(
    #         go.Scatter(y=self.interp_y, x=self.interp_x, mode="markers", name="Data")
    #     )
    #     fig.show()

    def kw_to_windspeed(self, df, kw_column="output_power_mean"):
        # Sampling a hundred points from the interpolated function
        # allows us to invert with an approximate accuracy of 12/100 or 0.1
        ws2 = np.linspace(0, 12, num=100)
        pc2 = self.powercurve_intrp(ws2)
        return df[kw_column].map(lambda x: ws2[np.abs(pc2 - x).argmin()])
