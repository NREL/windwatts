# Data Science & Core Logic

This section covers the `dw_tap` library and the scientific background of the data used in WindWatts.

## The `dw_tap` Package

`dw_tap` is the core functionality package containing logic for:
-   Height selection
-   Time interval selection
-   Spatial interpolation
-   Vertical interpolation
-   Wind rose calculations

### Setup

```bash
conda env create -f dw_tap/environment.yml
conda activate dw_tap
cd dw_tap
python setup.py install
```

### Testing

Run tests from the `dw_tap/` directory:

```bash
export PYTHONWARNINGS="ignore" && pytest --log-cli-level=INFO tests/
```

## Scientific Background: ERA5 Data

The WindWatts team evaluated the [ERA5 reanalysis dataset](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) for the US (2015-2023).

### Key Findings
-   **High temporal fidelity**: Correlation coeff of 0.775.
-   **Low error**: MAE of 1.58 m/s.
-   **Minimal bias**: Small positive bias of 0.09 m/s.

### Quantile-Based Aggregation

We use quantile-based aggregation to compress hourly data while preserving distribution shapes.

**Why Quantiles?**
-   Retains wind speed distribution shape (critical for non-linear energy output).
-   Reduces errors compared to 12x24 aggregation (0.1-2.6% vs 4-28%).
-   Efficient for storage and processing.

### Spatial Coverage

Currently focused on the continental United States.

![ERA5 Coverage](about/era5-coverage.png)

### Data Flow

![Data Flow](about/windwatts-data-flow.png)

