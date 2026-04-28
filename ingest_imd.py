import imdlib as imd
import os


def download_historical_rainfall():
    start_yr = 2021
    end_yr = 2021
    variable = 'rain'  # IMD variable for rainfall

    data_dir = './data/raw/imd'
    os.makedirs(data_dir, exist_ok=True)

    print(f"Fetching IMD {variable} data for {start_yr} (Ghatal flood timeframe)...")

    # Download the gridded data
    data = imd.get_data(variable, start_yr, end_yr, fn_format='yearwise')

    # For now, save it locally. Later we push this to Cloud Storage for the TPU.
    # We will slice out August 10-17 during the preprocessing phase.
    print(f"Download complete. Data cached for preprocessing.")


if __name__ == "__main__":
    download_historical_rainfall()