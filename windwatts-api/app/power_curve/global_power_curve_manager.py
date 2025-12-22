from pathlib import Path
from app.power_curve.power_curve_manager import PowerCurveManager

# Get the directory of this file and construct path to powercurves
_current_dir = Path(__file__).parent
power_curve_manager = PowerCurveManager(str(_current_dir / "powercurves"))