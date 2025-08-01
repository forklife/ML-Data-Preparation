# File: mt5_api.py
# Version: v1.3

import MetaTrader5 as mt5
import pandas as pd
from components.debug import log, ERROR, INFO
from components.version_control import check_version

# Определение file_path для текущего файла
file_path = __file__
# Проверка версии при запуске
check_version(file_path)

def connect_to_mt5():
    '''Initialize connection to MetaTrader 5 platform.'''
    if not mt5.initialize():
        log('Failed to initialize connection to MT5', ERROR)
        return False
    log('Connected to MT5 successfully', INFO)
    return True

def check_symbol_availability(symbol):
    '''Check if the symbol is available in MT5.'''
    available_symbols = mt5.symbols_get()
    available_symbol_names = [sym.name for sym in available_symbols]
    if symbol not in available_symbol_names:
        log(f'Symbol {symbol} is not available in MT5', ERROR)
        return False
    log(f'Symbol {symbol} is available', INFO)
    return True

def fetch_data(symbol, timeframe, start, end):
    '''Fetch historical data for a given symbol and timeframe.'''
    if not check_symbol_availability(symbol):
        return None
    
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None:
        log(f'Failed to fetch data for {symbol}', ERROR)
        return None
    
    log(f'Data fetched for {symbol} from {start} to {end}', INFO)
    
    # Преобразование в DataFrame и переименование tick_volume в volume
    data = pd.DataFrame(rates)
    if 'tick_volume' in data.columns:
        data.rename(columns={'tick_volume': 'volume'}, inplace=True)
        log("Column 'tick_volume' found and renamed to 'volume'.", INFO)
    
    return data

def disconnect_from_mt5():
    '''Shutdown MT5 connection properly.'''
    mt5.shutdown()
    log('Disconnected from MT5', INFO)
