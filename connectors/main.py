# File: main.py
# Version: v1.2

from connectors.mt5_api import connect_to_mt5, fetch_data, disconnect_from_mt5

def connect_and_fetch_data(symbol, timeframe, start, end):
    """Connect to MT5, fetch data for the specified symbol and timeframe, and then disconnect."""
    if not connect_to_mt5():
        print("[ERROR] Failed to connect to MT5")
        return None

    data = fetch_data(symbol, timeframe, start, end)
    disconnect_from_mt5()
    return data

# This script now provides utility functions for connecting to MT5 and fetching data in a simplified manner.
