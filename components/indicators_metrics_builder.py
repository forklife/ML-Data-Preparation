
# components/indicators_metrics_builder.py
# version: 0.30  ← +0.02 (binarized signals: rsi, dmi, macd, etc.)

import pandas as pd
import numpy as np
from numba import njit
import MetaTrader5 as mt5
import pandas_ta as ta  # Импортируем pandas-ta
from components.debug import log, DEBUG, ERROR, INFO
from components.version_control import check_version


def persistent_signal(signal_series):
    persistent = signal_series.copy()
    for i in range(1, len(signal_series)):
        if persistent.iloc[i] == 0 and persistent.iloc[i-1] == 1:
            if signal_series.iloc[i] == 0:
                persistent.iloc[i] = 0
            else:
                persistent.iloc[i] = 1
        elif persistent.iloc[i] == 0 and persistent.iloc[i-1] == 0:
            persistent.iloc[i] = 0
        elif persistent.iloc[i] == 1:
            if signal_series.iloc[i] == 0:
                persistent.iloc[i] = 0
            else:
                persistent.iloc[i] = 1
    return persistent


from connectors.main import connect_and_fetch_data
from components.data_transformation import initial_transform_data

file_path = __file__
check_version(file_path)

# === Главная функция-обертка для индикаторов ===
def calculate_indicators(raw_data, pbar=None):
    raw_data = raw_data.sort_values('datetime').reset_index(drop=True)
    index = raw_data.index  # Сохраняем исходный индекс

    high = raw_data['high']
    low = raw_data['low']
    close = raw_data['close']
    open_ = raw_data['open']
    volume = raw_data['volume']
    prev_close = close.shift(1)
    dt = raw_data['datetime']

    # Словарь для хранения новых колонок
    new_columns = {}

    # — Календарные признаки —
    new_columns['continuous_month'] = dt.dt.month
    new_columns['continuous_day_of_month'] = dt.dt.day
    new_columns['continuous_day_of_week'] = dt.dt.weekday
    new_columns['continuous_hour'] = dt.dt.hour
    new_columns['continuous_daytime_cycle_6h'] = new_columns['continuous_hour'] // 6
    new_columns['continuous_daytime_cycle_8h'] = new_columns['continuous_hour'] // 8
    new_columns['binary_weekends'] = (new_columns['continuous_day_of_week'] < 5).astype(int)  # Субботы/воскресенья = 0, будни = 1

    # Синусоидальные/косинусные кодировки
    new_columns['sincos_month_sin'] = np.sin(2 * np.pi * (new_columns['continuous_month'] - 1) / 12)
    new_columns['sincos_month_cos'] = np.cos(2 * np.pi * (new_columns['continuous_month'] - 1) / 12)
    dim = dt.dt.days_in_month
    new_columns['sincos_dom_sin'] = np.sin(2 * np.pi * (new_columns['continuous_day_of_month'] - 1) / dim)
    new_columns['sincos_dom_cos'] = np.cos(2 * np.pi * (new_columns['continuous_day_of_month'] - 1) / dim)
    new_columns['sincos_dow_sin'] = np.sin(2 * np.pi * new_columns['continuous_day_of_week'] / 7)
    new_columns['sincos_dow_cos'] = np.cos(2 * np.pi * new_columns['continuous_day_of_week'] / 7)
    new_columns['sincos_hour_sin'] = np.sin(2 * np.pi * new_columns['continuous_hour'] / 24)
    new_columns['sincos_hour_cos'] = np.cos(2 * np.pi * new_columns['continuous_hour'] / 24)
    new_columns['sincos_dc6h_sin'] = np.sin(2 * np.pi * new_columns['continuous_daytime_cycle_6h'] / 4)
    new_columns['sincos_dc6h_cos'] = np.cos(2 * np.pi * new_columns['continuous_daytime_cycle_6h'] / 4)
    new_columns['sincos_dc8h_sin'] = np.sin(2 * np.pi * new_columns['continuous_daytime_cycle_8h'] / 3)
    new_columns['sincos_dc8h_cos'] = np.cos(2 * np.pi * new_columns['continuous_daytime_cycle_8h'] / 3)
    new_columns['continuous_day_of_year'] = dt.dt.dayofyear
    diy = np.where(dt.dt.is_leap_year, 366, 365)
    new_columns['sincos_doy_sin'] = np.sin(2 * np.pi * (new_columns['continuous_day_of_year'] - 1) / diy)
    new_columns['sincos_doy_cos'] = np.cos(2 * np.pi * (new_columns['continuous_day_of_year'] - 1) / diy)
    new_columns['continuous_daytime_cycle'] = new_columns['continuous_hour'] // 6

    # — Корреляция валют —
    other_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
    corr_df = calculate_currency_correlation(raw_data, other_symbols, pbar=pbar)
    for col in corr_df.columns:
        new_columns[col] = corr_df[col]

    # — Историческая волатильность по дням и месяцам —
    new_columns['continuous_historical_volatility'] = calculate_historical_volatility(raw_data, pbar=pbar)

    # — Свечные паттерны (pandas-ta) —
    cdl_patterns = calculate_candlestick_patterns(raw_data, open_, high, low, close, pbar=pbar)
    log(f"Candlestick pattern columns: {cdl_patterns.columns.tolist()}", DEBUG)
    for col in cdl_patterns.columns:
        new_columns[f'binary_{col.lower()}_bull'] = (cdl_patterns[col] > 0).astype(int)
        new_columns[f'binary_{col.lower()}_bear'] = (cdl_patterns[col] < 0).astype(int)
        # log(f"Values in binary_{col.lower()}: {new_columns[f'binary_{col.lower()}'].unique().tolist()}", DEBUG)
        log(f"Values in binary_{col.lower()}_bull: {new_columns[f'binary_{col.lower()}_bull'].unique().tolist()}", DEBUG)
        log(f"Values in binary_{col.lower()}_bear: {new_columns[f'binary_{col.lower()}_bear'].unique().tolist()}", DEBUG)


    # — Классические индикаторы —
    new_columns['continuous_adx'] = calculate_adx(high, low, close, period=14, pbar=pbar)
    new_columns['continuous_rsi'] = calculate_rsi(close, period=14, pbar=pbar)
    new_columns['binary_rsi_signal_oversold'] = persistent_signal((new_columns['continuous_rsi'] < 40).astype(int))
    new_columns['binary_rsi_signal_overbought'] = persistent_signal((new_columns['continuous_rsi'] > 60).astype(int))
    new_columns['continuous_ema'] = calculate_ema_signal(close, period=14, pbar=pbar)
    new_columns['continuous_obv'] = calculate_obv_signal(close, volume, dt, pbar=pbar)
    new_columns['continuous_ad_line'] = calculate_ad_line_signal(high, low, close, volume, pbar=pbar)
    sma_short, sma_long = simple_moving_average_strategy(raw_data, short_window=10, long_window=20, pbar=pbar)
    new_columns['continuous_sma_short'] = sma_short
    new_columns['continuous_sma_long'] = sma_long

    fib_levels = calculate_fibonacci_retracement(high, low, window=96, pbar=pbar)
    for level in fib_levels.columns:
        new_columns[level] = fib_levels[level]

    double_top, double_bottom = identify_double_top_bottom(raw_data, min_distance=20, pbar=pbar)
    new_columns['continuous_double_top'] = double_top
    new_columns['continuous_double_bottom'] = double_bottom

    new_columns['continuous_aroon_oscillator'] = calculate_aroon_oscillator(high, low, period=25, pbar=pbar)
    macd, macd_signal, macd_hist = calculate_macd(close, fast_period=12, slow_period=26, signal_period=9, pbar=pbar)
    new_columns['continuous_macd'] = macd
    new_columns['continuous_macd_signal'] = macd_signal
    new_columns['continuous_macd_hist'] = macd_hist
    stoch, stoch_signal = calculate_stochastic_oscillator(high, low, close, period=14, pbar=pbar)
    new_columns['continuous_stoch_oscillator'] = stoch
    new_columns['continuous_stoch_signal'] = stoch_signal
    dmi_plus, dmi_minus = calculate_dmi(high, low, close, period=14, pbar=pbar)
    new_columns['continuous_dmi_plus'] = dmi_plus
    new_columns['continuous_dmi_minus'] = dmi_minus
    # Разделённые сигналы
    new_columns['binary_dmi_signal_bull'] = persistent_signal(((new_columns['continuous_dmi_plus'] > new_columns['continuous_dmi_minus']).astype(int)))
    new_columns['binary_dmi_signal_bear'] = persistent_signal(((new_columns['continuous_dmi_minus'] > new_columns['continuous_dmi_plus']).astype(int)))
    new_columns['continuous_parabolic_sar'] = calculate_parabolic_sar(high, low, close, pbar=pbar)

    # — Математика свечей —
    new_columns['continuous_candle_body'] = calculate_candle_body(open_, close, pbar=pbar)
    new_columns['continuous_upper_shadow'] = calculate_upper_shadow(high, open_, close, pbar=pbar)
    new_columns['continuous_lower_shadow'] = calculate_lower_shadow(low, open_, close, pbar=pbar)
    new_columns['continuous_price_range'] = calculate_price_range(high, low, pbar=pbar)
    new_columns['continuous_true_range'] = calculate_true_range(high, low, close, pbar=pbar)
    new_columns['continuous_price_change'] = calculate_price_change(close, prev_close, pbar=pbar)
    new_columns['continuous_percent_change'] = calculate_percent_change(close, prev_close, pbar=pbar)
    new_columns['continuous_close_open_ratio'] = calculate_close_open_ratio(close, open_, pbar=pbar)
    new_columns['binary_price_move_direction'] = calculate_price_move_direction(close, open_, pbar=pbar)
    new_columns['binary_candle_blue'] = (close > open_).astype(int)  # Бычья свеча: close > open
    new_columns['binary_candle_red'] = (close < open_).astype(int)   # Медвежья свеча: close < open
    log("Бычьи и медвежьи свечи рассчитаны (candle_blue, candle_red)", DEBUG)

    # — Блоки признаков —
    new_columns['continuous_open_close_gap'] = calculate_open_close_gap(raw_data, pbar=pbar)['open_close_gap']
    volume_features = calculate_volume_based_features(raw_data, pbar=pbar)
    new_columns['continuous_tick_volume_change'] = volume_features['tick_volume_change']
    new_columns['continuous_percent_tick_volume_change'] = volume_features['percent_tick_volume_change']
    new_columns['continuous_tick_volume_to_price_range_ratio'] = volume_features['tick_volume_to_price_range_ratio']
    new_columns['continuous_tick_volume_per_price_move'] = volume_features['tick_volume_per_price_move']
    new_columns['continuous_rolling_volume'] = volume_features['rolling_volume']
    trend_features = calculate_trend_based_features(raw_data, pbar=pbar)
    new_columns['continuous_rolling_mean_close'] = trend_features['rolling_mean_close']
    new_columns['continuous_rolling_std_close'] = trend_features['rolling_std_close']
    new_columns['continuous_rolling_max_high'] = trend_features['rolling_max_high']
    new_columns['continuous_rolling_min_low'] = trend_features['rolling_min_low']
    new_columns['continuous_rolling_median_close'] = trend_features['rolling_median_close']
    new_columns['continuous_rolling_price_change'] = trend_features['rolling_price_change']
    new_columns['continuous_rolling_range'] = trend_features['rolling_range']
    new_columns['continuous_rolling_price_direction'] = trend_features['rolling_price_direction']
    relative_features = calculate_absolute_relative_changes(raw_data, pbar=pbar)
    new_columns['continuous_relative_high'] = relative_features['relative_high']
    new_columns['continuous_relative_low'] = relative_features['relative_low']
    new_columns['continuous_relative_close'] = relative_features['relative_close']
    new_columns['continuous_high_low_ratio'] = relative_features['high_low_ratio']
    creative_features = calculate_creative_features(raw_data, pbar=pbar)
    new_columns['continuous_momentum'] = creative_features['momentum']
    new_columns['continuous_bollinger_upper'] = creative_features['bollinger_upper']
    new_columns['continuous_bollinger_lower'] = creative_features['bollinger_lower']
    new_columns['continuous_bollinger_range'] = creative_features['bollinger_range']
    new_columns['binary_volatility_compression'] = creative_features['volatility_compression']
    new_columns['binary_price_vs_volume_divergence'] = creative_features['price_vs_volume_divergence']
    new_columns['continuous_market_pressure'] = creative_features['market_pressure']
    pressure_features = calculate_price_pressure_features(raw_data, pbar=pbar)
    new_columns['sincos_upward_price_pressure'] = pressure_features['upward_price_pressure']
    new_columns['sincos_downward_price_pressure'] = pressure_features['downward_price_pressure']
    new_columns['continuous_intraday_volatility'] = calculate_intraday_volatility(raw_data, pbar=pbar)['intraday_volatility']

    # — Продвинутые индикаторы —
    obv_raw = volume * np.sign(close.diff().fillna(0))
    new_columns['continuous_obv_cum'] = obv_raw.cumsum()

    freq_min = (dt.diff().dt.total_seconds().dropna().mode().iloc[0] / 60
                if dt.diff().dt.total_seconds().dropna().size else 15)
    bars_per_day = int(round(1440 / freq_min))
    windows = {
        'week': bars_per_day * 5,
        'two_weeks': bars_per_day * 10,
        'month': bars_per_day * 21
    }
    for label, win in windows.items():
        new_columns[f'continuous_obv_{label}_rolling'] = obv_raw.rolling(win, 1).sum()

    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    new_columns['continuous_ad_clv'] = (clv.fillna(0) * volume).cumsum()

    new_columns['continuous_ema_14'] = close.ewm(span=14).mean()
    new_columns['continuous_ema_21'] = close.ewm(span=21).mean()
    new_columns['binary_macd_cross_up'] = persistent_signal((close > new_columns['continuous_ema_14']).astype(int))
    new_columns['binary_macd_cross_down'] = persistent_signal((close < new_columns['continuous_ema_14']).astype(int))

    new_columns['continuous_sma9'] = close.rolling(9, 1).mean()
    new_columns['continuous_sma26'] = close.rolling(26, 1).mean()
    new_columns['continuous_sma20'] = close.rolling(20, 1).mean()
    new_columns['continuous_sma50'] = close.rolling(50, 1).mean()

    new_columns['binary_aroon25_up'] = persistent_signal((new_columns['continuous_aroon_oscillator'] > 50).astype(int))
    new_columns['binary_aroon25_down'] = persistent_signal((new_columns['continuous_aroon_oscillator'] < -50).astype(int))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_hist = ema12 - ema26 - (ema12 - ema26).ewm(span=9).mean()

    low14 = low.rolling(14, 1).min()
    high14 = high.rolling(14, 1).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    new_columns['binary_stoch_signal_oversold'] = persistent_signal((stoch_k < 20).astype(int))
    new_columns['binary_stoch_signal_overbought'] = persistent_signal((stoch_k > 80).astype(int))



    # — Новые индикаторы —
    new_columns['continuous_atr'] = calculate_atr(high, low, close, period=14, pbar=pbar)
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (tp - sma_tp) / (0.015 * mad)
    new_columns['binary_cci_signal_low'] = persistent_signal((cci < -100).astype(int))
    new_columns['binary_cci_signal_high'] = persistent_signal((cci > 100).astype(int))

    
    
    # Исправлен расчет Super Trend (v0.30)
    atr = calculate_atr(high, low, close, period=10, pbar=pbar)
    hl2 = (high + low) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    super_trend = pd.Series(index=close.index, dtype=float)
    super_trend.iloc[0] = lower_band.iloc[0]
    for i in range(1, len(close)):
        if close.iloc[i-1] > super_trend.iloc[i-1]:
            super_trend.iloc[i] = lower_band.iloc[i]
        else:
            super_trend.iloc[i] = upper_band.iloc[i]


    # Исправлен расчет ADX (v0.30)
    plus_dm = (high - high.shift()).clip(lower=0)
    minus_dm = (low.shift() - low).clip(lower=0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(14).mean()

# === ДОБАВЛЕНЫ БИНАРНЫЕ СИГНАЛЫ (v0.30) ===
    new_columns['binary_super_trend_bull'] = persistent_signal((close > super_trend).astype(int))
    new_columns['binary_super_trend_bear'] = persistent_signal((close < super_trend).astype(int))

    new_columns['binary_adx_bull'] = persistent_signal(((adx > 25).astype(int)))
    new_columns['binary_adx_bear'] = persistent_signal(((adx > 25).astype(int)))

    sar_flip_diff = new_columns['continuous_parabolic_sar'].diff().abs() > 0
    new_columns['binary_sar_flip_up'] = (sar_flip_diff & (close > new_columns['continuous_parabolic_sar'])).astype(int)
    new_columns['binary_sar_flip_down'] = (sar_flip_diff & (close < new_columns['continuous_parabolic_sar'])).astype(int)

    new_columns['binary_ema14_price_cross_up'] = persistent_signal((close > new_columns['continuous_ema_14']).astype(int))
    new_columns['binary_ema14_price_cross_down'] = persistent_signal((close < new_columns['continuous_ema_14']).astype(int))

    new_columns['binary_ema21_price_cross_up'] = persistent_signal((close > new_columns['continuous_ema_21']).astype(int))
    new_columns['binary_ema21_price_cross_down'] = persistent_signal((close < new_columns['continuous_ema_21']).astype(int))

    new_columns['binary_sma9_26_cross_up'] = persistent_signal((new_columns['continuous_sma9'] > new_columns['continuous_sma26']).astype(int))
    new_columns['binary_sma9_26_cross_down'] = persistent_signal((new_columns['continuous_sma9'] < new_columns['continuous_sma26']).astype(int))

    new_columns['binary_sma20_50_cross_up'] = persistent_signal((new_columns['continuous_sma20'] > new_columns['continuous_sma50']).astype(int))
    new_columns['binary_sma20_50_cross_down'] = persistent_signal((new_columns['continuous_sma20'] < new_columns['continuous_sma50']).astype(int))


# Объединяем исходные и новые колонки
    new_columns_df = pd.DataFrame(new_columns, index=index)
    raw_data = pd.concat([raw_data, new_columns_df], axis=1)

    # Проверка NaN перед интерполяцией
    nan_count = raw_data.isna().sum().sum()
    if nan_count > 0:
        log(f"Индикаторы содержат NaN перед интерполяцией: {nan_count} всего.", INFO)
        # Интерполяция NaN для числовых колонок
        numeric_columns = raw_data.select_dtypes(include=[np.number]).columns
        raw_data[numeric_columns] = raw_data[numeric_columns].interpolate(method='linear').fillna(0)
        log("NaN в индикаторах интерполированы и оставшиеся заполнены нулями.", INFO)
        # Проверка NaN после интерполяции
        nan_count_after = raw_data.isna().sum().sum()
        log(f"Индикаторы содержат NaN после интерполяции: {nan_count_after} всего.", INFO)

    if pbar is not None:
        pbar.update(18)

    log("Все индикаторы успешно рассчитаны", DEBUG)
    raw_data.drop(
        columns=[
            'month', 'day_of_month', 'day_of_week', 'hour', 'daytime_cycle',
            'daytime_cycle_6h', 'daytime_cycle_8h'
        ],
        inplace=True,
        errors='ignore'
    )
    return raw_data

# === Функция для свечных паттернов ===
def calculate_candlestick_patterns(raw_data, open_, high, low, close, pbar=None):
    """
    Рассчитывает все свечные паттерны с использованием pandas-ta.
    Возвращает DataFrame с колонками, соответствующими паттернам (-100, 0, 100).
    """
    try:
        # Проверяем, что входные данные не содержат NaN или бесконечных значений
        input_data = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close
        })
        input_data = input_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        cdl_df = ta.cdl_pattern(open_=input_data['open'], high=input_data['high'], 
                               low=input_data['low'], close=input_data['close'], name='all')
        log("Свечные паттерны рассчитаны", DEBUG)
        if pbar is not None:
            pbar.update(1)
        return cdl_df
    except Exception as e:
        log(f"Ошибка при расчете свечных паттернов: {e}", ERROR)
        return pd.DataFrame()

# === Функции для новых индикаторов ===
def calculate_atr(high, low, close, period=14, pbar=None):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    if pbar is not None:
        pbar.update(1)
    log("ATR рассчитан", DEBUG)
    return atr

def calculate_super_trend_signal(high, low, close, period=10, multiplier=3, pbar=None):
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    super_trend = pd.Series(index=close.index, dtype=float)
    super_trend.iloc[0] = lower_band.iloc[0]
    for i in range(1, len(close)):
        if close.iloc[i-1] > super_trend.iloc[i-1]:
            super_trend.iloc[i] = lower_band.iloc[i]
        else:
            super_trend.iloc[i] = upper_band.iloc[i]
    signal = pd.Series(0, index=close.index)
    signal[close > super_trend] = 1   # Покупка
    signal[close < super_trend] = -1  # Продажа
    if pbar is not None:
        pbar.update(1)
    log("Сигнал Super Trend рассчитан", DEBUG)
    return signal

def calculate_adx_signal(high, low, close, period=14, pbar=None):
    plus_dm = (high - high.shift()).clip(lower=0)
    minus_dm = (low.shift() - low).clip(lower=0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(period).mean()
    signal = pd.Series(0, index=close.index)
    signal[(adx > 25) & (plus_di > minus_di)] = 1   # Покупка
    signal[(adx > 25) & (minus_di > plus_di)] = -1  # Продажа
    if pbar is not None:
        pbar.update(1)
    log("Сигнал ADX рассчитан", DEBUG)
    return signal

def calculate_cci_signal(high, low, close, period=20, pbar=None):
    tp = (high + low + close) / 3  # Типичная цена
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (tp - sma_tp) / (0.015 * mad)
    signal = pd.Series(0, index=close.index)
    signal[cci > 100] = -1   # Продажа
    signal[cci < -100] = 1   # Покупка
    if pbar is not None:
        pbar.update(1)
    log("Сигнал CCI рассчитан", DEBUG)
    return signal

def calculate_historical_volatility(raw_data, pbar=None):
    """
    Рассчитывает среднюю историческую волатильность для каждого календарного дня (месяц + день).
    Волатильность = |open - close| первого бара каждого дня, усредненная по годам.
    """
    df = raw_data.copy()
    df['date'] = df['datetime'].dt.date
    df['month_day'] = df['datetime'].dt.strftime('%m-%d')
    
    # Берем первый бар каждого дня
    daily_first = df.groupby('date').first().reset_index()
    daily_first['volatility'] = abs(daily_first['open'] - daily_first['close'])
    daily_first['year'] = pd.to_datetime(daily_first['date']).dt.year
    daily_first['month_day'] = pd.to_datetime(daily_first['date']).dt.strftime('%m-%d')
    
    # Усредняем волатильность по месяцам и дням для каждого года
    avg_volatility = daily_first.groupby(['month_day'])['volatility'].mean().reset_index()
    
    # Создаем маппинг для всех строк в исходном датасете
    volatility_map = dict(zip(avg_volatility['month_day'], avg_volatility['volatility']))
    result = df['month_day'].map(volatility_map).fillna(0)
    
    if pbar is not None:
        pbar.update(1)
    log("Историческая волатильность по дням и месяцам рассчитана", DEBUG)
    return result

# === Корреляция валют ===
def calculate_currency_correlation(raw_data, other_symbols, period=96, pbar=None):
    start_date = raw_data['datetime'].min().to_pydatetime()
    end_date = raw_data['datetime'].max().to_pydatetime()
    timeframe = mt5.TIMEFRAME_M15
    xau_pct = raw_data['close'].pct_change()
    correlations = {}
    for symbol in other_symbols:
        raw_other = connect_and_fetch_data(symbol, timeframe, start_date, end_date)
        if raw_other is None or len(raw_other) == 0:
            log(f"Не удалось получить данные для {symbol}", ERROR)
            continue
        raw_other = pd.DataFrame(raw_other)
        transformed_other = initial_transform_data(raw_other, freq='15min')
        if transformed_other is None:
            continue
        transformed_other = transformed_other.set_index('datetime')
        raw_data_aligned = raw_data.set_index('datetime')
        aligned_other, aligned_xau = transformed_other['close'].align(raw_data_aligned['close'], join='inner')
        other_pct = aligned_other.pct_change()
        corr = other_pct.rolling(period).corr(aligned_xau.pct_change())
        correlations[f'corr_{symbol}'] = corr.reindex(raw_data['datetime']).fillna(0)
    if pbar:
        pbar.update(1)
    log("Корреляция валют рассчитана", INFO)
    return pd.DataFrame(correlations, index=raw_data.index)

# === Простые помощники для свечной математики ===
def calculate_candle_body(open_, close, pbar=None):
    log("Тело свечи рассчитано", DEBUG)
    return close - open_

def calculate_upper_shadow(high, open_, close, pbar=None):
    log("Верхняя тень рассчитана", DEBUG)
    return high - np.maximum(open_, close)

def calculate_lower_shadow(low, open_, close, pbar=None):
    log("Нижняя тень рассчитана", DEBUG)
    return np.minimum(open_, close) - low

def calculate_price_range(high, low, pbar=None):
    log("Диапазон цен рассчитан", DEBUG)
    return high - low

def calculate_true_range(high, low, close, pbar=None):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(tr1)
    log("Истинный диапазон рассчитан", DEBUG)
    return true_range

def calculate_price_change(close, prev_close, pbar=None):
    log("Изменение цены рассчитано", DEBUG)
    return close - prev_close

def calculate_percent_change(close, prev_close, pbar=None):
    log("Процентное изменение рассчитано", DEBUG)
    return ((close - prev_close) / prev_close * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

def calculate_close_open_ratio(close, open_, pbar=None):
    log("Отношение закрытия к открытию рассчитано", DEBUG)
    return (close / open_).replace([np.inf, -np.inf], np.nan).fillna(0)

def calculate_price_move_direction(close, open_, pbar=None):
    log("Направление движения цены рассчитано", DEBUG)
    return (close > open_).astype(int)

# === Признаки разрыва и объема ===
def calculate_open_close_gap(raw_data, pbar=None):
    date_only = raw_data['datetime'].dt.date
    last_close = raw_data.groupby(date_only)['close'].last().shift(1)
    first_open = raw_data.groupby(date_only)['open'].first()
    result = pd.DataFrame({'open_close_gap': first_open - last_close}, index=raw_data.index)
    result = result.reindex(raw_data.index, method='ffill')
    log("Разрыв открытия-закрытия рассчитан", DEBUG)
    return result

def calculate_volume_based_features(raw_data, pbar=None):
    result = raw_data.copy()
    for col in ['volume', 'high', 'low', 'close', 'open']:
        result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)

    date_only = result['datetime'].dt.date
    result['tick_volume_change'] = result.groupby(date_only)['volume'].diff().fillna(0).clip(-1e6, 1e6)
    prev_vol = result.groupby(date_only)['volume'].shift(1)
    result['percent_tick_volume_change'] = (result['tick_volume_change'] / prev_vol * 100)\
        .replace([np.inf, -np.inf], np.nan).fillna(0)

    rng = (result['high'] - result['low']).replace(0, np.nan)
    result['tick_volume_to_price_range_ratio'] = (result['volume'] / rng)\
        .replace([np.inf, -np.inf], np.nan).fillna(0)

    move = (result['close'] - result['open']).abs().replace(0, np.nan)
    result['tick_volume_per_price_move'] = (result['volume'] / move)\
        .replace([np.inf, -np.inf], np.nan).fillna(0)

    result['rolling_volume'] = result['volume'].rolling(window=9).mean().fillna(0)
    log("Признаки на основе объема рассчитаны", DEBUG)
    return result[['tick_volume_change', 'percent_tick_volume_change', 
                   'tick_volume_to_price_range_ratio', 'tick_volume_per_price_move', 
                   'rolling_volume']]

def calculate_trend_based_features(raw_data, pbar=None):
    result = raw_data.copy()
    result['rolling_mean_close'] = result['close'].rolling(window=9).mean()
    result['rolling_std_close'] = result['close'].rolling(window=9).std()
    result['rolling_max_high'] = result['high'].rolling(window=9).max()
    result['rolling_min_low'] = result['low'].rolling(window=9).min()
    result['rolling_median_close'] = result['close'].rolling(window=9).median()
    result['rolling_price_change'] = result['close'].diff().rolling(window=9).mean()
    result['rolling_range'] = (result['high'] - result['low']).rolling(window=9).mean()
    result['rolling_price_direction'] = (result['close'] > result['open']).astype(int).rolling(9).mean().fillna(0)
    log("Признаки на основе трендов рассчитаны", DEBUG)
    return result[['rolling_mean_close', 'rolling_std_close', 'rolling_max_high', 
                   'rolling_min_low', 'rolling_median_close', 'rolling_price_change', 
                   'rolling_range', 'rolling_price_direction']]

def calculate_absolute_relative_changes(raw_data, pbar=None):
    result = raw_data.copy()
    result['relative_high'] = result['high'].pct_change().fillna(0) * 100
    result['relative_low'] = result['low'].pct_change().fillna(0) * 100
    result['relative_close'] = result['close'].pct_change().fillna(0) * 100
    result['high_low_ratio'] = (result['high'] / result['low']).replace([np.inf, -np.inf], np.nan).fillna(0)
    log("Абсолютные и относительные изменения рассчитаны", DEBUG)
    return result[['relative_high', 'relative_low', 'relative_close', 'high_low_ratio']]

def calculate_creative_features(raw_data, pbar=None):
    result = raw_data.copy()
    result['momentum'] = result['close'].diff(9)

    sma20 = result['close'].rolling(20).mean()
    std20 = result['close'].rolling(20).std()
    result['bollinger_upper'] = sma20 + std20 * 2
    result['bollinger_lower'] = sma20 - std20 * 2
    result['bollinger_range'] = result['bollinger_upper'] - result['bollinger_lower']
    result['volatility_compression'] = (result['bollinger_range'] <
                                       result['bollinger_range'].rolling(20).mean()).astype(int)

    result['price_vs_volume_divergence'] = (
        ((result['close'] > result['open']).astype(int) !=
         (result['volume'] > result['volume'].shift(1)).astype(int))
    ).astype(int)

    result['market_pressure'] = np.where(
        result['close'] > result['open'],
        result['close'] * result['volume'],
        0
    ) - np.where(
        result['close'] < result['open'],
        result['close'] * result['volume'],
        0
    )
    log("Креативные признаки рассчитаны", DEBUG)
    return result[['momentum', 'bollinger_upper', 'bollinger_lower', 
                   'bollinger_range', 'volatility_compression', 
                   'price_vs_volume_divergence', 'market_pressure']]

def calculate_price_pressure_features(raw_data, pbar=None):
    result = raw_data.copy()
    result['upward_price_pressure'] = (result['close'] > result['open']).rolling(12).mean().fillna(0)
    result['downward_price_pressure'] = (result['close'] < result['open']).rolling(12).mean().fillna(0)
    log("Признаки давления цены рассчитаны", DEBUG)
    return result[['upward_price_pressure', 'downward_price_pressure']]

def calculate_intraday_volatility(raw_data, pbar=None):
    date_only = raw_data['datetime'].dt.date
    open_on_day = raw_data.groupby(date_only)['open'].first()
    avg_price = (raw_data['open'] + raw_data['high'] + raw_data['low']) / 3
    open_mapped = date_only.map(open_on_day)

    log(f"avg_price dtype: {avg_price.dtype}", DEBUG)
    log(f"open_mapped dtype: {open_mapped.dtype}", DEBUG)

    if not np.issubdtype(open_mapped.dtype, np.number):
        log("Ошибка: open_mapped содержит нечисловые значения", ERROR)
        open_mapped = pd.to_numeric(open_mapped, errors='coerce').fillna(0)

    vol = np.where(avg_price > open_mapped,
                   raw_data['high'] - open_mapped,
                   raw_data['low'] - open_mapped)
    result = pd.DataFrame({'intraday_volatility': np.nan_to_num(vol)}, index=raw_data.index)
    log("Внутридневная волатильность рассчитана", DEBUG)
    return result

def calculate_adx(high, low, close, period=14, pbar=None):
    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()

    if pbar is not None:
        pbar.update(1)
    log("ADX рассчитан", DEBUG)
    return pd.Series(adx, index=high.index).fillna(0)

def calculate_rsi(series, period=14, pbar=None):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    if pbar is not None:
        pbar.update(1)
    log("RSI рассчитан", DEBUG)
    return pd.Series(rsi, index=series.index)

def calculate_ema_signal(series, period=14, pbar=None):
    ema = series.ewm(span=period, adjust=False).mean()
    if pbar is not None:
        pbar.update(1)
    log("EMA рассчитан", DEBUG)
    return pd.Series(ema, index=series.index)

def calculate_obv_signal(close, volume, datetime, pbar=None):
    if not (close.index.equals(volume.index) and close.index.equals(datetime.index)):
        raise ValueError("Индексы Series 'close', 'volume' и 'datetime' должны совпадать.")
    date_only = datetime.dt.date
    obv = pd.Series(index=close.index, dtype=float)
    for date, group in close.groupby(date_only):
        delta = group.diff()
        sign = np.sign(delta)
        obv_day = (sign * volume.loc[group.index]).fillna(0).cumsum()
        obv.loc[group.index] = obv_day
    if pbar is not None:
        pbar.update(1)
    log("OBV рассчитан с ежедневным сбросом", DEBUG)
    return obv

def calculate_ad_line_signal(high, low, close, volume, pbar=None):
    clv = ((close - low) - (high - close)) / (high - low)
    ad_line = clv.fillna(0).cumsum()
    if pbar is not None:
        pbar.update(1)
    log("Линия A/D рассчитана", DEBUG)
    return pd.Series(ad_line, index=close.index)

def simple_moving_average_strategy(raw_data, short_window=10, long_window=20, pbar=None):
    s = raw_data['close']
    short_ma = s.rolling(short_window, 1).mean()
    long_ma = s.rolling(long_window, 1).mean()
    if pbar is not None:
        pbar.update(1)
    log("Простая скользящая средняя рассчитана", DEBUG)
    return pd.Series(short_ma, index=raw_data.index), pd.Series(long_ma, index=raw_data.index)

def calculate_fibonacci_retracement(high, low, window=96, pbar=None):
    max_price = high.rolling(window, 1).max()
    min_price = low.rolling(window, 1).min()
    diff = max_price - min_price
    levels = pd.DataFrame(index=high.index)
    levels['continuous_level_0236'] = max_price - 0.236 * diff
    levels['continuous_level_0382'] = max_price - 0.382 * diff
    levels['continuous_level_0618'] = max_price - 0.618 * diff
    levels['continuous_level_0786'] = max_price - 0.786 * diff
    if pbar is not None:
        pbar.update(1)
    log("Уровни Фибоначчи рассчитаны", DEBUG)
    return levels

def find_patterns(close, max_idx, min_idx, min_distance, tolerance):
    n = len(close)
    double_top = np.zeros(n)
    double_bottom = np.zeros(n)

    cum_min = np.minimum.accumulate(close[::-1])[::-1]
    cum_max = np.maximum.accumulate(close[::-1])[::-1]

    for i in range(len(max_idx) - 1):
        for j in range(i + 1, len(max_idx)):
            if max_idx[j] - max_idx[i] >= min_distance:
                if abs(close[max_idx[i]] - close[max_idx[j]]) / close[max_idx[i]] <= tolerance:
                    neckline = np.min(close[max_idx[i]:max_idx[j]])
                    if cum_min[max_idx[j]] < neckline:
                        double_top[max_idx[j]] = close[max_idx[j]]
                        break

    for i in range(len(min_idx) - 1):
        for j in range(i + 1, len(min_idx)):
            if min_idx[j] - min_idx[i] >= min_distance:
                if abs(close[min_idx[i]] - close[min_idx[j]]) / close[min_idx[i]] <= tolerance:
                    neckline = np.max(close[min_idx[i]:min_idx[j]])
                    if cum_max[min_idx[j]] > neckline:
                        double_bottom[min_idx[j]] = close[min_idx[j]]
                        break
    return double_top, double_bottom

def identify_double_top_bottom(raw_data, min_distance=20, tolerance=0.01, pbar=None):
    close = raw_data['close'].values
    dt_series = pd.Series(np.zeros(len(raw_data)), index=raw_data.index, dtype=float)
    db_series = dt_series.copy()

    local_max = raw_data['close'] == raw_data['close'].rolling(min_distance * 2 + 1, center=True).max()
    local_min = raw_data['close'] == raw_data['close'].rolling(min_distance * 2 + 1, center=True).min()
    tops, bottoms = find_patterns(
        close,
        np.where(local_max)[0],
        np.where(local_min)[0],
        min_distance,
        tolerance
    )
    dt_series[:] = tops
    db_series[:] = bottoms
    if pbar is not None:
        pbar.update(1)
    log("Двойные вершины и основания определены", DEBUG)
    return dt_series, db_series

def calculate_aroon_oscillator(high, low, period=25, pbar=None):
    aroon_up = 100 * (period - high.rolling(period).apply(lambda x: period - 1 - x.argmax(), raw=True)) / period
    aroon_down = 100 * (period - low.rolling(period).apply(lambda x: period - 1 - x.argmin(), raw=True)) / period
    aroon = aroon_up - aroon_down
    if pbar is not None:
        pbar.update(1)
    log("Осциллятор Aroon рассчитан", DEBUG)
    return pd.Series(aroon, index=high.index)

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9, pbar=None):
    exp1 = series.ewm(span=fast_period, adjust=False).mean()
    exp2 = series.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd - macd_signal
    if pbar is not None:
        pbar.update(1)
    log("MACD рассчитан", DEBUG)
    return pd.Series(macd, index=series.index), pd.Series(macd_signal, index=series.index), pd.Series(macd_hist, index=series.index)

def calculate_stochastic_oscillator(high, low, close, period=14, pbar=None):
    low_min = low.rolling(period).min()
    high_max = high.rolling(period).max()
    stoch = 100 * (close - low_min) / (high_max - low_min)
    stoch_signal = stoch.rolling(9).mean()
    if pbar is not None:
        pbar.update(1)
    log("Стохастический осциллятор рассчитан", DEBUG)
    return pd.Series(stoch, index=close.index), pd.Series(stoch_signal, index=close.index)

def calculate_dmi(high, low, close, period=14, pbar=None):
    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    if pbar is not None:
        pbar.update(1)
    log("DMI рассчитан", DEBUG)
    return pd.Series(plus_di, index=high.index), pd.Series(minus_di, index=high.index)

def calculate_parabolic_sar(high, low, close, start=0.02, increment=0.02, maximum=0.2, pbar=None):
    sar = close.copy()
    trend = 1
    af = start
    ep = high.iloc[0]

    sar.iloc[0] = low.iloc[0]
    for i in range(1, len(close)):
        sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
        if trend == 1:
            if low.iloc[i] < sar.iloc[i]:
                trend = -1
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + increment, maximum)
        else:
            if high.iloc[i] > sar.iloc[i]:
                trend = 1
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = start
            else:
                if low.iloc[i] < sar.iloc[i]:
                    ep = low.iloc[i]
                    af = min(af + increment, maximum)
    if pbar is not None:
        pbar.update(1)
    log("Параболический SAR рассчитан", DEBUG)
    return pd.Series(sar, index=close.index)