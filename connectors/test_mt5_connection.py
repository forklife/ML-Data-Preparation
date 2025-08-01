# File: test_mt5_connection.py

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def connect_to_mt5():
    '''Initialize connection to MetaTrader 5 platform.'''
    if not mt5.initialize():
        print("[ERROR] Failed to initialize connection to MT5")
        return False
    print("[INFO] Connected to MT5 successfully")
    return True

def disconnect_from_mt5():
    '''Shutdown MT5 connection properly.'''
    mt5.shutdown()
    print("[INFO] Disconnected from MT5")

def check_symbol_availability(symbol):
    '''Check if the symbol is available in MT5.'''
    available_symbols = mt5.symbols_get()
    available_symbol_names = [sym.name for sym in available_symbols]
    if symbol not in available_symbol_names:
        print(f"[ERROR] Symbol {symbol} is not available in MT5")
        return False
    print(f"[INFO] Symbol {symbol} is available")
    return True

def fetch_data_with_copy_rates_range(symbol, timeframe, start, end):
    '''Fetch historical data for a given symbol and timeframe using copy_rates_range.'''
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None:
        print(f"[ERROR] Failed to fetch data for {symbol} using copy_rates_range")
        return None
    print(f"[INFO] Data fetched for {symbol} using copy_rates_range from {start} to {end}")
    
    # Преобразуем данные в DataFrame
    data = pd.DataFrame(rates)
    data['datetime'] = pd.to_datetime(data['time'], unit='s')
    data.drop(columns=['time'], inplace=True)
    return data

def main():
    # Настройки для тестирования
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_H1  # Пример: часовой таймфрейм
    start_date = datetime(2023, 12, 1)
    end_date = datetime(2023, 12, 31)
    
    # Подключаемся к MT5 и загружаем данные
    if not connect_to_mt5():
        return

    # Проверка доступности символа
    if not check_symbol_availability(symbol):
        disconnect_from_mt5()
        return

    # Получение данных с использованием copy_rates_range
    rates_data = fetch_data_with_copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates_data is not None:
        print("[INFO] Data sample using copy_rates_range:")
        print(rates_data.head())
    else:
        print("[ERROR] No data fetched using copy_rates_range.")

    # Отключаемся от MT5
    disconnect_from_mt5()

if __name__ == "__main__":
    main()
