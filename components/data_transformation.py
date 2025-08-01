# File: components/data_transformation.py
# Version: v2.1  (added explicit NaN logging)

import pandas as pd
from components.debug import log, DEBUG, ERROR, INFO
from components.version_control import check_version
import numpy as np

file_path = __file__
check_version(file_path)

def transform_data(indicators, raw_data, symbol, TF, start_date, end_date):
    """Объединение индикаторов с сырыми данными и проверка на корректность."""
    log("Transforming and merging data...", INFO)
    
    try:
        # Добавляем колонки symbol и timeframe
        raw_data['date'] = raw_data['datetime'].dt.date
        raw_data['symbol'] = symbol
        raw_data['timeframe'] = TF
        
        # Фильтруем данные по нужной начальной дате
        raw_data = raw_data[raw_data['datetime'] >= start_date]

        # Проверка, что индикаторы не пустые
        if indicators is None or indicators.empty:
            log("Indicators are missing or empty.", ERROR)
            return None
        
        # Проверка наличия колонки 'datetime' в индикаторах
        if 'datetime' not in indicators.columns:
            log("Adding 'datetime' column to indicators.", INFO)
            indicators['datetime'] = raw_data['datetime'].values
        
        # Логируем перед объединением
        log(f"Columns in raw data before merge: {', '.join(raw_data.columns)}", INFO)
        log(f"Columns in indicators before merge: {', '.join(indicators.columns)}", INFO)

        # Объединение данных с индикаторами по дате и времени
        merged_data = pd.merge(raw_data, indicators, on='datetime', how='left', suffixes=('', '_y'))
        
        # Удаляем дубликаты колонок, оставляя только нужные
        columns_to_drop = [col for col in merged_data.columns if '_y' in col]
        merged_data.drop(columns=columns_to_drop, inplace=True)

        # Удаление ненужных колонок после объединения
        columns_to_remove = ['time', 'symbol', 'real_volume', 'real_volume_x', 'date', 'timeframe', 'isRealData']
        merged_data.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        # Логируем финальный набор колонок после объединения и очистки
        log(f"Columns after cleaning duplicates and renaming: {', '.join(merged_data.columns)}", INFO)

        # Проверка на наличие NaN и inf значений
        nan_count = merged_data.isna().sum().sum()
        log(f"Data contains NaN values before interpolation: {nan_count} total.", INFO)
        # Линейная интерполяция для числовых колонок
        numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
        merged_data[numeric_columns] = merged_data[numeric_columns].interpolate(method='linear').fillna(0)
        log("Interpolated NaN values and filled remaining with 0.", INFO)
        # Логирование NaN после интерполяции
        nan_count_after = merged_data.isna().sum().sum()
        log(f"Data contains NaN values after interpolation: {nan_count_after} total.", INFO)
        
        inf_count = np.isinf(merged_data).sum().sum()
        if inf_count > 0:
            log(f"Data contains infinite values: {inf_count} total.", INFO)
            merged_data.replace([np.inf, -np.inf], 0, inplace=True)
            log("Replaced infinite values with 0.", INFO)
        
        # Возвращаем итоговый DataFrame
        return merged_data
    except Exception as e:
        log(f"Error during data transformation: {e}", ERROR)
        return None

def initial_transform_data(raw_data, freq, pbar=None):
    """Начальная трансформация сырых данных перед расчетом индикаторов."""
    try:
        # Шаг 1: Преобразование колонки 'time' в формат datetime до минут и переименование в 'datetime', удаление 'time'
        raw_data['datetime'] = pd.to_datetime(raw_data['time'], unit='s', errors='coerce').dt.floor(freq)
        raw_data.drop(columns=['time'], inplace=True)
        log(f"Converted 'time' to 'datetime' with {freq} precision and removed 'time' column.", INFO)

        # Шаг 2: Сортировка данных по 'datetime'
        raw_data = raw_data.sort_values('datetime').reset_index(drop=True)
        log("Sorted data by 'datetime'.", INFO)
        original_len = len(raw_data)

        # Шаг 3: Удаление суббот и воскресений
        initial_row_count = len(raw_data)
        raw_data = raw_data[raw_data['datetime'].dt.weekday < 5].reset_index(drop=True)

        # Шаг 4: Построение полного диапазона дат с учетом частоты
        full_dates = pd.date_range(start=raw_data['datetime'].min(), end=raw_data['datetime'].max(), freq=freq)
        log(f"Generated full date range from {full_dates.min()} to {full_dates.max()} with {len(full_dates)} total rows based on freq '{freq}'.", INFO)

        # Шаг 5: Создание полного диапазона дат и заполнение пропущенных дат
        raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
        date_range = pd.date_range(start=raw_data['datetime'].min(), end=raw_data['datetime'].max(), freq=freq)
        raw_data = raw_data.set_index('datetime').reindex(date_range)
        raw_data = raw_data.ffill().reset_index().rename(columns={'index': 'datetime'})
        log(f"Filled missing dates. Total rows after reindexing: {len(raw_data)}", INFO)
        added_rows_count = len(raw_data) - original_len
        log(f"Synthetic rows added: {added_rows_count}", DEBUG)

        # Шаг 6: Интерполяция числовых колонок
        if raw_data.isnull().values.any():
            numeric_columns = raw_data.select_dtypes(include=[np.number]).columns
            raw_data[numeric_columns] = raw_data[numeric_columns].interpolate(method='linear').fillna(0)
            log("Interpolated NaN values in raw data and filled remaining with 0.", INFO)

        # Шаг 7: Проверка и переименование колонки 'volume'
        if 'volume' not in raw_data.columns:
            if 'tick_volume' in raw_data.columns:
                raw_data.rename(columns={'tick_volume': 'volume'}, inplace=True)
                log("Column 'tick_volume' found and renamed to 'volume'.", INFO)
            else:
                log("Missing 'volume' and 'tick_volume' columns in raw data.", ERROR)
                return None

        # Обновление прогресс-бара, если он передан
        if pbar is not None:
            pbar.update(1)

        # Шаг 8: Возврат обновленного DataFrame
        log("Initial data transformation completed successfully.", INFO)
        return raw_data

    except Exception as e:
        log(f"Error during initial data transformation: {e}", ERROR)
        return None