import pandas as pd
import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import njit

# Загрузка твоего датасета
df = pd.read_csv('dataset_XAUUSD_M15_patterns.csv', parse_dates=['datetime'])

# Настройки
bear_patterns = [col for col in df.columns if 'cdl_cdl_' in col and '_bear' in col]
bull_patterns = [col for col in df.columns if 'cdl_cdl_' in col and '_bull' in col]

# Исключение паттернов с нулевыми значениями
original_bear_count = len(bear_patterns)
original_bull_count = len(bull_patterns)

bear_patterns = [p for p in bear_patterns if df[p].sum() > 0]
bull_patterns = [p for p in bull_patterns if df[p].sum() > 0]

excluded_bear = original_bear_count - len(bear_patterns)
excluded_bull = original_bull_count - len(bull_patterns)

print(f'Всего bear паттернов: {original_bear_count}, исключено с 0 результатами: {excluded_bear}')
print(f'Всего bull паттернов: {original_bull_count}, исключено с 0 результатами: {excluded_bull}')

# Параметры анализа
move_required = 8.0
window_after_signal = 8
confirmation_window = 5
num_cores = 100

# Оптимизированная функция анализа одиночного паттерна
@njit
def analyze_single_pattern_numba(close_prices, extreme_prices, signal_indices, window_after_signal, move_required, direction):
    total_signals, successful_signals, movements = 0, 0, []

    for idx in signal_indices:
        entry_price = close_prices[idx]
        if idx + 1 + window_after_signal > len(extreme_prices):
            continue

        if direction == 0:  # bear
            extreme_future_price = np.min(extreme_prices[idx+1 : idx+1+window_after_signal])
            movement = entry_price - extreme_future_price
        else:  # bull
            extreme_future_price = np.max(extreme_prices[idx+1 : idx+1+window_after_signal])
            movement = extreme_future_price - entry_price

        total_signals += 1

        if movement >= move_required:
            successful_signals += 1
            movements.append(movement)

    accuracy = successful_signals / total_signals if total_signals else 0
    avg_movement = np.mean(np.array(movements)) if movements else 0
    return accuracy, avg_movement, total_signals

# Анализ одиночных паттернов

def analyze_single_pattern(pattern, direction='bear'):
    signal_indices = np.where(df[pattern].values == 1)[0]
    close_prices = df['close'].values
    extreme_prices = df['low'].values if direction == 'bear' else df['high'].values

    direction_code = 0 if direction == 'bear' else 1

    accuracy, avg_movement, total_signals = analyze_single_pattern_numba(
        close_prices, extreme_prices, signal_indices, window_after_signal, move_required, direction_code)

    if total_signals > 0:
        freq_per_month = total_signals / df['datetime'].dt.to_period('M').nunique()
        return {
            'pattern': pattern,
            'accuracy': accuracy,
            'average_movement': avg_movement,
            'monthly_frequency': freq_per_month
        }
    return None

# Анализ одиночных паттернов
print(f'Всего одиночных bear паттернов после фильтрации: {len(bear_patterns)}')
single_bear_results = Parallel(n_jobs=num_cores)(delayed(analyze_single_pattern)(pattern, 'bear') for pattern in tqdm(bear_patterns, mininterval=0))
single_bear_results_df = pd.DataFrame([res for res in single_bear_results if res is not None])

print(f'Всего одиночных bull паттернов после фильтрации: {len(bull_patterns)}')
single_bull_results = Parallel(n_jobs=num_cores)(delayed(analyze_single_pattern)(pattern, 'bull') for pattern in tqdm(bull_patterns, mininterval=0))
single_bull_results_df = pd.DataFrame([res for res in single_bull_results if res is not None])

# Рассчитываем среднюю точность одиночных паттернов
mean_accuracy_bear = single_bear_results_df['accuracy'].mean()
mean_accuracy_bull = single_bull_results_df['accuracy'].mean()

# Оставляем для комбинаций только паттерны выше средней точности
filtered_bear_patterns = single_bear_results_df[single_bear_results_df['accuracy'] > mean_accuracy_bear]['pattern'].tolist()
filtered_bull_patterns = single_bull_results_df[single_bull_results_df['accuracy'] > mean_accuracy_bull]['pattern'].tolist()

print(f'Оставлено bear паттернов для комбинаций: {len(filtered_bear_patterns)}')
print(f'Оставлено bull паттернов для комбинаций: {len(filtered_bull_patterns)}')

# Оптимизированная функция анализа комбинаций паттернов
@njit
def analyze_pattern_combo_numba(close_prices, extreme_prices, pattern_matrix, primary_idx, confirm_idxs, confirmation_window, window_after_signal, move_required, direction):
    total_combos, successful_combos, combo_movements = 0, 0, []

    for idx in range(pattern_matrix.shape[0] - confirmation_window - window_after_signal):
        if pattern_matrix[idx, primary_idx]:
            for offset in range(1, confirmation_window + 1):
                if np.all(pattern_matrix[idx + offset, confirm_idxs]):
                    entry_price = close_prices[idx + offset]

                    if direction == 0:  # bear
                        extreme_future_price = np.min(extreme_prices[idx + offset + 1: idx + offset + 1 + window_after_signal])
                        movement = entry_price - extreme_future_price
                    else:  # bull
                        extreme_future_price = np.max(extreme_prices[idx + offset + 1: idx + offset + 1 + window_after_signal])
                        movement = extreme_future_price - entry_price

                    total_combos += 1

                    if movement >= move_required:
                        successful_combos += 1
                        combo_movements.append(movement)
                    break

    accuracy = successful_combos / total_combos if total_combos else 0
    avg_movement = np.mean(np.array(combo_movements)) if combo_movements else 0
    return accuracy, avg_movement, total_combos

# Анализ комбинаций паттернов

def analyze_pattern_combo(combo, direction='bear'):
    pattern_matrix = df[list(combo)].values
    close_prices = df['close'].values
    extreme_prices = df['low'].values if direction == 'bear' else df['high'].values

    direction_code = 0 if direction == 'bear' else 1

    accuracy, avg_movement, total_combos = analyze_pattern_combo_numba(
        close_prices, extreme_prices, pattern_matrix, 0, np.arange(1, len(combo)),
        confirmation_window, window_after_signal, move_required, direction_code)

    if total_combos > 0:
        combo_freq_month = total_combos / df['datetime'].dt.to_period('M').nunique()
        return {
            'pattern_combo': ' + '.join(combo),
            'accuracy': accuracy,
            'average_movement': avg_movement,
            'monthly_frequency': combo_freq_month
        }
    return None

# Анализ комбинаций паттернов (2, 3 и 4)
combo_results = []

for direction, patterns in [('bear', filtered_bear_patterns), ('bull', filtered_bull_patterns)]:
    for combo_length in [2, 3, 4]:
        pattern_combos = list(combinations(patterns, combo_length))
        print(f'Всего комбинаций {direction} паттернов из {combo_length}: {len(pattern_combos)}')

        results = Parallel(n_jobs=num_cores)(delayed(analyze_pattern_combo)(combo, direction) for combo in tqdm(pattern_combos, mininterval=0))
        combo_results.extend([res for res in results if res is not None])

combo_results_df = pd.DataFrame(combo_results)

# Сохраняем результаты
single_bear_results_df.to_csv('single_bear_patterns_analysis.csv', index=False)
single_bull_results_df.to_csv('single_bull_patterns_analysis.csv', index=False)
combo_results_df.to_csv('combo_patterns_analysis.csv', index=False)