# components/main.py
# Version: v1.4  (removed add_advanced_indicators)

from components.data_transformation import transform_data
from components.data_validation import validate_dataset
from components.indicators_metrics_builder import calculate_indicators
from components.debug import log, INFO, ERROR
from components.version_control import check_version
import pandas as pd

file_path = __file__
check_version(file_path)

def run_data_processing(data, timeframe="M15"):
    """
    Full ETL pipeline for raw data.
    Returns DataFrame with all metrics, or None on error.
    """
    try:
        log("Starting data transformation process", INFO)
        transformed_data = transform_data(data)
        log("Data transformation completed", INFO)

        log("Starting data validation process", INFO)
        if not validate_dataset(transformed_data):
            log("Data validation failed", ERROR)
            return None
        log("Data validation passed", INFO)

        log("Starting indicators calculation", INFO)
        indicators = calculate_indicators(transformed_data)
        log("Indicators calculated", INFO)

        return indicators

    except Exception as e:
        log(f"Error in main coordination: {str(e)}", ERROR)
        return None