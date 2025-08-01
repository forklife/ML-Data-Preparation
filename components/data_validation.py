
# File: components/data_validation.py
# Version: v0.11  (added validation for candle_blue and candle_red)

import pandas as pd
import numpy as np
from components.debug import log, INFO, ERROR, DEBUG
from components.version_control import check_version

file_path = __file__
check_version(file_path)

def validate_dataset(dataset):
    """
    Проверяет корректность датасета: наличие ключевых колонок, типы данных, 
    отсутствие недопустимых значений и корректность диапазонов индикаторов.
    Возвращает True, если датасет валиден, иначе False.
    """
    log("Validating dataset...", INFO)
    
    try:
        # Проверка наличия ключевых колонок
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in dataset.columns]
        if missing_columns:
            log(f"Missing required columns: {', '.join(missing_columns)}", ERROR)
            return False

        # Проверка наличия хотя бы одного свечного паттерна
        cdl_columns = [col for col in dataset.columns if col.startswith('binary_cdl_')]
        if not cdl_columns:
            log("No candlestick pattern columns found in dataset.", ERROR)
            return False
        log(f"Found {len(cdl_columns)} candlestick pattern columns: {cdl_columns}", DEBUG)

        # Проверка типов данных
        if not pd.api.types.is_datetime64_any_dtype(dataset['datetime']):
            log("Column 'datetime' is not in datetime format.", ERROR)
            return False
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(dataset[col]):
                log(f"Column '{col}' is not numeric.", ERROR)
                return False

        # Проверка на наличие NaN
        nan_count = dataset.isna().sum().sum()
        if nan_count > 0:
            log(f"Dataset contains {nan_count} NaN values.", ERROR)
            return False

        # Проверка на наличие бесконечных значений
        inf_count = np.isinf(dataset.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            log(f"Dataset contains {inf_count} infinite values.", ERROR)
            return False

        # Проверка диапазонов индикаторов
        # indicator_checks = {
            # 'rsi_signal': lambda x: x.isin([-1, 0, 1]),
            # 'dmi_signal': lambda x: x.isin([-1, 0, 1]),
            # 'aroon25_signal': lambda x: x.isin([-1, 0, 1]),
            # 'macd_cross_signal': lambda x: x.isin([-1, 0, 1]),
            # 'stoch_signal': lambda x: x.isin([-1, 0, 1]),
            # 'sar_flip_signal': lambda x: x.isin([-1, 0, 1]),
            # 'ema14_price_cross': lambda x: x.isin([-1, 0, 1]),
            # 'ema21_price_cross': lambda x: x.isin([-1, 0, 1]),
            # 'sma9_26_cross': lambda x: x.isin([-1, 0, 1]),
            # 'sma20_50_cross': lambda x: x.isin([-1, 0, 1]),
            # 'volatility_compression': lambda x: x.isin([0, 1]),
            # 'price_vs_volume_divergence': lambda x: x.isin([0, 1]),
            # 'price_move_direction': lambda x: x.isin([0, 1]),
            # 'candle_blue': lambda x: x.isin([0, 1]),  # Бычья свеча
            # 'candle_red': lambda x: x.isin([0, 1]),  # Медвежья свеча
            # 'super_trend_signal': lambda x: x.isin([-1, 0, 1]),
            # 'adx_signal': lambda x: x.isin([-1, 0, 1]),
            # 'cci_signal': lambda x: x.isin([-1, 0, 1]),
            # 'weekends': lambda x: x.isin([0, 1]),
            # 'historical_volatility': lambda x: (x >= 0)  # Волатильность должна быть положительной
        # }

        # # Проверка свечных паттернов
        # for col in cdl_columns:
            # indicator_checks[col] = lambda x: x.isin([0, 1])

        # corr_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
        # for symbol in corr_symbols:
            # col = f'corr_{symbol}'
            # if col in dataset.columns:
                # indicator_checks[col] = lambda x: (x >= -1) & (x <= 1)

        # for col, check in indicator_checks.items():
            # if col in dataset.columns:
                # if not check(dataset[col]).all():
                    # invalid_values = dataset[col][~check(dataset[col])].unique()
                    # log(f"Invalid values in '{col}': {invalid_values.tolist()}. Expected range: [-1, 0, 1] or [0, 1] for binary, or [-1, 1] for correlations.", ERROR)
                    # return False

        # Проверка диапазона RSI (0–100), если колонка существует
        if 'rsi' in dataset.columns:
            if not (dataset['rsi'].between(0, 100)).all():
                log("Invalid values in 'rsi': values must be between 0 and 100.", ERROR)
                return False

        # Проверка положительных значений для цен
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (dataset[col] <= 0).any():
                log(f"Invalid values in '{col}': values must be positive.", ERROR)
                return False

        # Проверка логической целостности: high >= low, close <= high, close >= low
        if (dataset['high'] < dataset['low']).any():
            log("Invalid data: 'high' must be >= 'low'.", ERROR)
            return False
        if (dataset['close'] > dataset['high']).any() or (dataset['close'] < dataset['low']).any():
            log("Invalid data: 'close' must be between 'low' and 'high'.", ERROR)
            return False

        log("Dataset validation successful.", INFO)
        return True

    except Exception as e:
        log(f"Error during dataset validation: {e}", ERROR)
        return False