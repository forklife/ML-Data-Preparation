# dataset_builder.py
# Version: v1.0  (removed add_advanced_indicators, updated tqdm)

import os
from datetime import datetime
import MetaTrader5 as mt5
from components.main import run_data_processing
from connectors.main import connect_and_fetch_data
from components.data_validation import validate_dataset
from components.data_transformation import transform_data, initial_transform_data
from components.indicators_metrics_builder import calculate_indicators
from components.debug import log, INFO, ERROR
from components.version_control import check_version
import pandas as pd
from tqdm import tqdm

file_path = __file__
check_version(file_path)

OUTPUT_FOLDER = 'output-datasets-fortraining'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

symbol_string = "XAUUSD"
symbols = [s.strip() for s in symbol_string.split(',')]

timeframe = mt5.TIMEFRAME_M15
start_date = datetime(2021, 1, 1)
end_date = datetime(2025, 12, 31)

for symbol in symbols:
    try:
        log("Connecting to data source...", INFO)
        log(f"Symbol: {symbol}", INFO)
        log(f"timeframe: {timeframe}", INFO)

        # --- fetch ---
        with tqdm(total=1, desc=f"Fetching Data for {symbol}", unit="step") as pbar:
            raw_data = connect_and_fetch_data(symbol, timeframe, start_date, end_date)
            raw_data = pd.DataFrame(raw_data)
            pbar.update(1)

        if raw_data.empty:
            log(f"Fetched data for {symbol} is empty.", ERROR)
            raise Exception("No data fetched")

        print("Список колонок для:", symbol)
        print(raw_data.columns.tolist())

        if 'volume' not in raw_data.columns:
            log(f"Volume column is missing after fetching data for {symbol}.", ERROR)
            raise Exception("Missing volume column")

        # --- initial transform ---
        log("Initial transforming data...", INFO)
        with tqdm(total=1, desc=f"Initial Transformation {symbol}", unit="step") as pbar:
            transformed_raw_data = initial_transform_data(raw_data, freq='15min', pbar=pbar)
            pbar.update(1)

        # --- indicators ---
        log("Calculating indicators...", INFO)
        with tqdm(total=18, desc=f"Calculating Indicators {symbol}", unit="indicator") as pbar:
            indicators = calculate_indicators(transformed_raw_data, pbar=pbar)

        # --- post-transform & merge ---
        log("Transforming data after indicators calculation...", INFO)
        with tqdm(total=1, desc=f"Post-Indicators Transformation {symbol}", unit="step") as pbar:
            transformed_data = transform_data(
                indicators,
                transformed_raw_data,
                symbol,
                timeframe,
                start_date,
                end_date
            )
            # --- exclude unwanted columns ---
            cols_to_exclude = ["adx", "rsi", "ema", "obv", "ad_line", "sma_short", "sma_long"]
            transformed_data = transformed_data.drop(
                columns=[c for c in cols_to_exclude if c in transformed_data.columns],
                errors="ignore"
            )
            log(f"Excluded columns {cols_to_exclude} from dataset", INFO)
            log(f"Final columns after exclusion: {', '.join(transformed_data.columns)}", INFO)
            pbar.update(1)

        if transformed_data is None:
            raise Exception("Transformation failed")

        # --- validation ---
        log("Validating dataset...", INFO)
        with tqdm(total=1, desc=f"Validating Data {symbol}", unit="step") as pbar:
            if validate_dataset(transformed_data):
                log(f"Data validation passed for {symbol}", INFO)
            else:
                log(f"Data validation failed for {symbol}", ERROR)
                raise Exception("Validation failed")
            pbar.update(1)

        # --- save ---
        save_path = os.path.join(OUTPUT_FOLDER, f"dataset_{symbol}_M{timeframe}.csv")
        transformed_data.to_csv(save_path, index=False)
        log(f"Dataset saved to {save_path} for {symbol}", INFO)

    except Exception as e:
        log(f"Error during dataset building for {symbol}: {str(e)}", ERROR)