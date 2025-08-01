#  █████   VERSION 2   █████
#  Добавлен блок для сохранения и чтения метрик из файла metrics_prev.json

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    "max_split_size_mb:2096,"              # Максимальный размер сплит-блока 2 ГБ  4096 2048 1024 512 256
    # "garbage_collection_threshold:0.9,"    # При ~80% фрагментированной памяти начинать «уборку»
    # "expandable_segments:True,"           # Позволяет выделять дополнительные крупные сегменты
    # "pinned_use_cuda_host_register:True,"  # Включение pinned-memory (если ROCm это поддерживает)
    # "pinned_num_register_threads:8,"
    # "pinned_use_background_threads:True"
)

# Устанавливаем нужные переменные окружения
# os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
# os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import torch.distributed as dist
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.ticker import AutoLocator
import traceback
from datetime import timedelta
from pytorch_lightning.callbacks import EarlyStopping, Callback
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from darts import TimeSeries
from darts import concatenate
from darts.utils.missing_values import fill_missing_values
from darts.utils.losses import MapeLoss, MAELoss, SmapeLoss
from darts.explainability.tft_explainer import TFTExplainer
from torchmetrics import MetricCollection, MeanSquaredError, R2Score, MeanAbsolutePercentageError, SymmetricMeanAbsolutePercentageError
from darts.dataprocessing.transformers import (
    Scaler,
    MissingValuesFiller,
    Mapper,
    InvertibleMapper,
)
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
from darts.utils.callbacks import TFMProgressBar
from darts.models import (
    TFTModel,
    # TCNModel,
    # NBEATSModel,
    # RNNModel,
    # NHiTSModel,
    # TransformerModel,
)
from darts.metrics import mape, rmse, mae, r2_score, mse, rmsse, smape
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import (
    QuantileRegression,
    GaussianLikelihood,
    LaplaceLikelihood,
    ExponentialLikelihood,
)
import warnings
import time
import requests
import json  # ДОБАВЛЕНО для работы с JSON-файлом



warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set PyTorch settings

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')
torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
tran = nn.Transformer(batch_first=True)  # No warning
# fork_rng(devices=range(torch.cuda.device_count()))

# Custom Loss Functions
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse_loss(y_pred, y_true))
        return loss

class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        mean_true = torch.mean(y_true)
        ss_tot = torch.sum((y_true - mean_true) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return -r2  # Negative because we want to maximize R2

@rank_zero_only
def print_once(*args, **kwargs):
    print(*args, **kwargs)
    
@rank_zero_only
def run_prediction(my_model, horizon, val_cov_predict, future_covariates_full_transformed):
    # Генерация предсказаний
    pred_series = my_model.predict(
        n=horizon,
        past_covariates=val_cov_predict,
        future_covariates=future_covariates_full_transformed,
        show_warnings=True,
        num_samples=100,
        verbose=True,
    )
    return pred_series

@rank_zero_only
def print_sorted_importances(importances: dict):
    """
    Для каждого ключа словаря importances нормализует значения так,
    чтобы их сумма была 100, сортирует их по убыванию и выводит в формате "название: значение"
    (значения округлены до целых чисел).
    """
    for key, df in importances.items():
        print(f"\n{key}:")
        if df.empty:
            print("Empty DataFrame")
        else:
            # Предполагаем, что DataFrame состоит из одной строки
            series = df.iloc[0]
            total = series.sum()
            # Нормализуем так, чтобы сумма значений была 100
            normalized = series
            sorted_series = normalized.sort_values(ascending=False)
            for col, val in sorted_series.items():
                print(f"{col}: {val}")

def save_model(model, model_folder, file_name):
    """
    Saves the model to the specified directory with a unique filename.
    """
    model_path = os.path.join(model_folder, file_name)
    model.save(model_path)
    print(f"DEBUG: Model saved to {model_path}")

def load_darts_model_from_pth(model_class, model_path, input_chunk_length, output_chunk_length, device='cpu'):
    """
    Загружает модель Darts из .pth.tar файла.
    
    :param model_class: Класс модели Darts (например, TCNModel).
    :param model_path: Путь к файлу модели (.pth.tar).
    :param input_chunk_length: Длина входного окна, использованная при обучении.
    :param output_chunk_length: Длина выходного окна, использованная при обучении.
    :param device: Устройство для загрузки модели ('cpu' или 'cuda').
    :return: Загруженная модель или None, если загрузка не удалась.
    """
    if not os.path.exists(model_path):
        print(f"ERROR: Файл модели по пути {model_path} не существует.")
        return None

    try:
        # Инициализируем модель с необходимыми параметрами
        model = model_class(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            # Добавьте другие параметры, если они использовались при обучении
        )
        
        # Загружаем состояние модели
        checkpoint = torch.load(model_path, map_location=device)
        
        # Проверяем структуру сохранённого файла
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Загружаем state_dict в внутреннюю модель Darts
        model.model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        print(f"DEBUG: Модель успешно загружена из {model_path}")
        return model
    except Exception as e:
        print(f"ERROR: Не удалось загрузить модель из {model_path}. Причина: {e}")
        return None

def color_text(value, ranges, is_r2=False):
    """
    Returns colored text based on the value and defined ranges.
    Formats the value to four decimal places.
    """
    formatted_value = f"{value:.4f}"  # Format to four decimal places

    if is_r2:
        # Logic for R2: Green if >= first range, Yellow if >= second, else Red
        if value >= ranges[0]:
            return f"\033[1;32m{formatted_value}\033[0m"  # Green
        elif value >= ranges[1]:
            return f"\033[1;33m{formatted_value}\033[0m"  # Yellow
        else:
            return f"\033[1;31m{formatted_value}\033[0m"  # Red
    else:
        # Logic for other metrics: Green if <= first range, Yellow if <= second, else Red
        if value <= ranges[0]:
            return f"\033[1;32m{formatted_value}\033[0m"  # Green
        elif value <= ranges[1]:
            return f"\033[1;33m{formatted_value}\033[0m"  # Yellow
        else:
            return f"\033[1;31m{formatted_value}\033[0m"  # Red           

def get_freq_from_filename(filename):
    """
    Determines the frequency of the time series data based on the filename.
    """
    freq_map = {
        'Daily': 'D',
        'Weekly': 'W',
        'H12': '12h',
        'H8': '8h',
        'H4': '4h',
        'H3': '3h',
        'H2': '2h',
        'H1': '1h',
        'M30': '30min',
        'M15': '15min',
        'M10': '10min',
        'M5': '5min',
        'M1': 'min'
    }
    for key, value in freq_map.items():
        if key in filename:
            return value
    raise ValueError(f"DEBUG: Unsupported timeframe in filename: {filename}")

#  █████   ДОБАВЛЕННАЯ ФУНКЦИЯ   █████
# Используем глобальную переменную для хранения предыдущих метрик
prev_metrics = {}

def track_metric_changes(rmse_val, mae_val, mape_val, r2_val):
    """
    Вычисляет дельту изменений метрик относительно предыдущего вызова.
    Если предыдущие метрики отсутствуют, функция возвращает пустые строки.
    Также сохраняет новые метрики в файл metrics_prev.json.
    """
    global prev_metrics
    prev_file = "metrics_prev.json"
    # Если глобальная переменная пуста, пытаемся загрузить из файла (если он существует)
    if not prev_metrics and os.path.isfile(prev_file):
        try:
            with open(prev_file, "r", encoding="utf-8") as f:
                prev_metrics = json.load(f)
        except Exception as e:
            print(f"WARNING: Не удалось загрузить предыдущие метрики: {e}")
            prev_metrics = {}
    
    # Если предыдущих метрик нет, сохраняем текущие и возвращаем пустые дельты
    if not prev_metrics:
        prev_metrics = {"RMSE": rmse_val, "MAE": mae_val, "MAPE": mape_val, "R2": r2_val}
        with open(prev_file, "w", encoding="utf-8") as f:
            json.dump(prev_metrics, f, ensure_ascii=False, indent=4)
        return "", "", "", ""
    
    def delta_str(current, previous):
        return f"{(current - previous):+0.4f}"

    rmse_delta = delta_str(rmse_val, prev_metrics.get("RMSE", rmse_val))
    mae_delta = delta_str(mae_val, prev_metrics.get("MAE", mae_val))
    mape_delta = delta_str(mape_val, prev_metrics.get("MAPE", mape_val))
    r2_delta = delta_str(r2_val, prev_metrics.get("R2", r2_val))
    
    # Обновляем глобальные метрики и сохраняем их
    prev_metrics = {"RMSE": rmse_val, "MAE": mae_val, "MAPE": mape_val, "R2": r2_val}
    with open(prev_file, "w", encoding="utf-8") as f:
        json.dump(prev_metrics, f, ensure_ascii=False, indent=4)
    
    return rmse_delta, mae_delta, mape_delta, r2_delta

def process_file(file_name, clean_data_folder, model_folder):
    """
    Processes a single CSV file: loading, preprocessing, training, and evaluation.
    """
    try:
        # Determine terminal width for formatting
        terminal_width = shutil.get_terminal_size().columns
        line = '-' * terminal_width
        linestar = '☆' * terminal_width

        # Load the data using pandas
        file_path = os.path.join(clean_data_folder, file_name)
        data = pd.read_csv(file_path, low_memory=False)

        print_once("\nDATA LOADING, PROCESSING, VALIDATION BEFORE START")
        print_once(f"{line}\n")

        # Determine frequency from filename
        freq = get_freq_from_filename(file_name)
        print_once(f"DEBUG: Data Frequency: {freq}")
        
        # Преобразуем строковый формат в datetime
        data['datetime'] = pd.to_datetime(data['datetime'])

        # Фильтруем данные, оставляя только начиная с 1 января 2023 года
        # data = data[data['datetime'] >= '2023-01-09 01:00:00']        
        data = data[(data['datetime'] >= '2025-01-01 01:15:00') & (data['datetime'] <= '2025-12-28 02:30:00')]         

        # Convert to TimeSeries and fill missing dates
        series = TimeSeries.from_dataframe(
            data, 
            time_col='datetime',  
            value_cols=["close"], #'close', # ["open", "high", "low", "close"]
            fill_missing_dates=True,
            freq=freq
        ).astype(np.float32)
        print_once("DEBUG: Target Series created from Data Frame - Successfully")

        # Fill missing values in the target series
        series = fill_missing_values(series, method="linear")
        print_once("DEBUG: Target Series Filled with Missing Values - Successfully")

        # Prepare covariate data
        data2 = data
        print_once("DEBUG: Covariative Data Loaded from Main Target File - Successfully")

        # Define feature groups
        
        # data2_columns = [col for col in data2 if col not in ['close','datetime']]
        data2_columns = [  

    "obv_week_rolling",
    "bollinger_range",
    "aroon_oscillator",
    "macd",
    "obv_two_weeks_rolling",
    "cdl_cdl_spinningtop",
    "aroon25_signal",
    "volume",
    "rolling_max_high",
    "level_0382",
    "obv_month_rolling",
    "cci_signal",
    "sma20_50_cross",
    "tick_volume_per_price_move",
    "open",
    "low",
    "high",
    "volatility_compression",
    "intraday_volatility",
    "price_vs_volume_divergence",
    "ema14_price_cross",
    "sma9_26_cross",
    "ema21_price_cross",
    "stoch_oscillator",
    "ema_14",
    "stoch_signal",
    "cdl_cdl_matchinglow",
    "atr",
    "historical_volatility",
    "cdl_cdl_closingmarubozu",
    "double_top",
    "candle_blue",
    "cdl_cdl_longline",
    "downward_price_pressure",
    "macd_signal",
    "dmi_minus"
]
        
        
        # Convert to TimeSeries and fill missing dates
        covariate_series = TimeSeries.from_dataframe(
            data2, 
            time_col='datetime', 
            value_cols=data2_columns, 
            fill_missing_dates=True,
            freq=freq
        ).astype(np.float32)
        
        print_once("DEBUG: Covariates Series created from Data Frame - Successfully")
        
        # Fill missing values in the covariate_series series
        covariate_series = fill_missing_values(covariate_series, method="linear")
        print_once("DEBUG: Covariate Series Filled with Missing Values - Successfully")
        
        
        # future_cov = pd.read_csv("combined_fullcast_result_15min.csv2", parse_dates=["datetime"])
        
        # future_cov["dummy_column"] = 0
        
        # freq = '15min'
        
        # excluded_cols = ["id_col","datetime"]
        # excluded_substrings = ["close","datetime","AUDUSD","EURUSD","NZDUSD","USDCAD","USDJPY","GBPUSD"]

        # cols_for_cov = [
            # c for c in future_cov.columns
            # if c not in excluded_cols
            # and not any(sub in c for sub in excluded_substrings)
        # ]


        # future_covariates = TimeSeries.from_dataframe(
            # future_cov,
            # time_col='datetime',
            # value_cols=cols_for_cov,
            # fill_missing_dates=True,  # Darts «дополнит» пропущенные метки времени 
            # freq=freq
        # ).astype(np.float32)
        

        # print_once("\n\n Колонки в future_covariates: \n\n", future_covariates.components)
        
        # # Fill missing values in the covariate_series series
        # future_covariates = fill_missing_values(future_covariates, method="linear")
        
        # required_start_time = series.start_time()
        # required_end_time = series.end_time()
        # print_once(f"required_start_time: {required_start_time}  required_end_time: {required_end_time}")
        # future_covariates_full = future_covariates

        # # # # Ensure sliced data starts with required time and ends in the future
        # future_covariates = future_covariates.slice(required_start_time, required_end_time)
        

        

         
        # Calculate the split number for train-validation split
        splitnumber = 0.80
        
        # # Split the target series into training and validation sets
        # train_future_covariates, val_future_covariates = future_covariates.split_after(splitnumber)
        # print_once("DEBUG: Main Series split into Train/Validation Sets - Successfully ")

        # Split the target series into training and validation sets
        train_series, val_series = series.split_after(splitnumber)
        print_once("DEBUG: Main Series split into Train/Validation Sets - Successfully ")
        
        # Split the covariate series into training and validation sets
        train_cov_series, val_cov_series = covariate_series.split_after(splitnumber)
        print_once("DEBUG: Covariative Series split into Train/Validation Sets - Successfully ")

        # plt.figure(figsize=(60, 9))



        # train_series["close"].plot(label="Real Price", color='black', linewidth=1)
        # train_future_covariates["XAUUSD_Bar50"].plot(label="Predicted", linewidth=1)



        # ax = plt.gca()
        # # ax.yaxis.set_major_locator(MultipleLocator(20))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        # plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=2))
        # plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
        # plt.grid(visible=True, which='major', color='grey', linestyle='-', linewidth=0.5)
        # plt.grid(visible=True, which='minor', color='lightgrey', linestyle='--', linewidth=0.5)
        # plt.xticks(rotation=90)
        # ax_right = ax.twinx()
        # ax_right.set_ylim(ax.get_ylim())
        # # ax_right.yaxis.set_major_locator(MultipleLocator(20))
        # ax_right.set_ylabel("Values (duplicated axis)")
        # plt.title(
            # f"train_series vs train_future_covariates Zoomed"
        # )
        # plt.xlabel("Time")
        # plt.ylabel("Values")
        # plt.tight_layout()

        # timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        # plot_filename = f"data-check.png"
        # plt.savefig(os.path.join(model_folder, plot_filename))
        # plt.close()
        # print_once(f"\033[32m  DEBUG: runpodctl send models/{plot_filename}\033[0m")

        print_once("DEBUG: Trying to normalize data - !!! HOLD ON !!!")
        
        # Scale the target series
        scaler_series = Scaler()
        train_transformed = scaler_series.fit_transform(train_series)       
        val_transformed = scaler_series.transform(val_series)
        series_transformed = scaler_series.transform(series)
        
        print_once("DEBUG: Normalization for series is finished - Successfully")
        
        # Scale the  covariate  series
        scaler_covariate_series = Scaler()
        train_cov_transformed = scaler_covariate_series.fit_transform(train_cov_series)       
        val_cov_transformed = scaler_covariate_series.transform(val_cov_series)
        covariate_series_transformed = scaler_covariate_series.transform(covariate_series)   
        
        print_once("DEBUG: Normalization for covariate_series is finished - Successfully")

        # FUTURE COVARIATES
        future_cov_columns = [
            "weekends", "month_sin", "month_cos", "dom_sin", "dom_cos",
            "dow_sin", "dow_cos", "hour_sin", "hour_cos",
            "dc6h_sin", "dc6h_cos", "dc8h_sin"
        ]

        future_covariates = TimeSeries.from_dataframe(
            data,
            time_col='datetime',
            value_cols=future_cov_columns,
            fill_missing_dates=True,
            freq=freq
        ).astype(np.float32)

        future_covariates = fill_missing_values(future_covariates, method="linear")

        train_future_covariates, val_future_covariates = future_covariates.split_after(splitnumber)
        scaler_future_covariates = Scaler()
        train_future_covariates_transformed = scaler_future_covariates.fit_transform(train_future_covariates)
        val_future_covariates_transformed = scaler_future_covariates.transform(val_future_covariates)
        future_covariates_full_transformed = scaler_future_covariates.transform(future_covariates)


        # Scale the  future_covariates  series
        # scaler_future_covariates = Scaler()
        # train_future_covariates_transformed = scaler_future_covariates.fit_transform(train_future_covariates)       
        # val_future_covariates_transformed = scaler_future_covariates.transform(val_future_covariates)
        # future_covariates_transformed = scaler_future_covariates.transform(future_covariates)   
        # future_covariates_full_transformed = scaler_future_covariates.transform(future_covariates_full)   
        
        print_once("DEBUG: Normalization for future_covariates is finished - Successfully")

        print_once(f"{line}\n")  # Adds an empty line

        print_once("DEBUG: FINAL CHECKS, KEY INFORMATION")
        print_once(f"{line}\n")  # Adds an empty line

        # Output date ranges for verification
        print_once(f"Date range for full series:", series.time_index.min(), "to", series.time_index.max())
        print_once(f"Date range for training set train_transformed:", train_transformed.time_index.min(), "to", train_transformed.time_index.max())
        print_once(f"Date range for validation set val_transformed:", val_transformed.time_index.min(), "to", val_transformed.time_index.max())
        print_once(f"Date range for training covariates train_cov_transformed:", train_cov_transformed.time_index.min(), "to", train_cov_transformed.time_index.max())
        print_once(f"Date range for validation covariates val_cov_transformed:", val_cov_transformed.time_index.min(), "to", val_cov_transformed.time_index.max())
        # print_once(f"Date range for training future covariates val_cov_transformed:", train_future_covariates_transformed.time_index.min(), "to", train_future_covariates_transformed.time_index.max())
        # print_once(f"Date range for validation future val_cov_transformed:", val_future_covariates_transformed.time_index.min(), "to", val_future_covariates_transformed.time_index.max())
        print_once(f"\n")

        print_once(f"Date range for full covariates:", covariate_series.time_index.min(), "to", covariate_series.time_index.max())
        print_once(f"Date range for training covariates train_cov_transformed:", train_cov_transformed.time_index.min(), "to", train_cov_transformed.time_index.max())
        print_once(f"Date range for validation covariates val_cov_transformed:", val_cov_transformed.time_index.min(), "to", val_cov_transformed.time_index.max())
        print_once(f"\n")

        # Visual validation of covariate series
        print_once("\nDEBUG: covariate_series_scaled Head Visual Validation")
        print_once("\n", covariate_series_transformed.to_dataframe().head())
        print_once("\nDEBUG: Head Visual Validation")

        # Check the number of columns in training and validation datasets for series and covariates
        print_once(f"Number of columns in train_transformed:", train_transformed.n_components)
        print_once(f"Number of columns in val_transformed:", val_transformed.n_components)
        print_once(f"Number of columns in train_cov_transformed:", train_cov_transformed.n_components)
        print_once(f"Number of columns in val_cov_transformed:", val_cov_transformed.n_components)
        print_once(f"\n")

        # Check the sizes of training and validation sets
        print_once(f"Sizes of train_transformed (samples, timesteps):", train_transformed.n_samples, train_transformed.n_timesteps)
        print_once(f"Sizes of val_transformed (samples, timesteps):", val_transformed.n_samples, val_transformed.n_timesteps)
        print_once(f"Sizes of train_cov_transformed (samples, timesteps):", train_cov_transformed.n_samples, train_cov_transformed.n_timesteps)
        print_once(f"Sizes of val_cov_transformed (samples, timesteps):", val_cov_transformed.n_samples, val_cov_transformed.n_timesteps)
        print_once(f"\n")

        # Check for missing values in the data
        print_once(f"Are there NaNs in train_transformed:", train_transformed.to_dataframe().isnull().values.any())
        print_once(f"Are there NaNs in val_transformed:", val_transformed.to_dataframe().isnull().values.any())
        print_once(f"Are there NaNs in train_cov_transformed:", train_cov_transformed.to_dataframe().isnull().values.any())
        print_once(f"Are there NaNs in val_cov_transformed:", val_cov_transformed.to_dataframe().isnull().values.any())
        print_once(f"\n")

        # Check for negative values in the data
        print_once(f"Are there negative values in train_transformed:", (train_transformed.to_dataframe().lt(0)).values.any())
        print_once(f"Are there negative values in val_transformed:", (val_transformed.to_dataframe().lt(0)).values.any())
        print_once(f"Are there negative values in train_cov_transformed:", (train_cov_transformed.to_dataframe().lt(0)).values.any())
        print_once(f"Are there negative values in val_cov_transformed:", (val_cov_transformed.to_dataframe().lt(0)).values.any())
        
        df_val_cov_transformed = val_cov_transformed.to_dataframe()
        
        negative_cols = df_val_cov_transformed.columns[df_val_cov_transformed.lt(0).any()].tolist()
        if negative_cols:
            print_once(f"Колонки с отрицательными значениями: {negative_cols}")
        else:
            print_once("Нет колонок с отрицательными значениями.")
        
        print_once(f"############################################################ \n")

        # Display the first few rows of each series
        print_once(f"\n train_transformed", train_transformed.to_dataframe().head())
        print_once(f"\n val_transformed", val_transformed.to_dataframe().head())
        print_once(f"\n val_cov_transformed", val_cov_transformed.to_dataframe().head())
        print_once(f"\n train_cov_transformed", train_cov_transformed.to_dataframe().head())

        # List of time series for validation
        series_list = [train_transformed, train_cov_transformed, val_transformed, val_cov_transformed]

        # Check each time series for NaN or inf
        for i, series_obj in enumerate(series_list):
            dftest = series_obj.to_dataframe()
            nan_count = dftest.isna().sum().sum()
            inf_count = np.isinf(dftest.values).sum()
            print_once(f"TimeSeries {i+1}: NaN count = {nan_count}, inf count = {inf_count}")

        # Check for consistency of time indices between series and covariates
        print_once(f"Do time indices match in train_transformed and train_cov_transformed:", 
              train_transformed.time_index.equals(train_cov_transformed.time_index))
        print_once(f"Do time indices match in val_transformed and val_cov_transformed:", 
              val_transformed.time_index.equals(val_cov_transformed.time_index))
        print_once(f"{line}\n")  # Adds an empty line

        # Define DataLoader keyword arguments
        dataloader_kwargs = {
            "drop_last": False,  # Do not drop the last incomplete batch
            "pin_memory": True,  # Do not pin memory
            "num_workers": 2,
            "persistent_workers": True,
            "prefetch_factor": 2,
        }

        # Define early stopping callback
        early_stop_callback = EarlyStopping(
            monitor="val_loss",  # Monitor validation loss
            patience=6,
            mode='min',
            verbose=True
        )
        
        
        script_name = datetime.datetime.now().strftime("%Y-%m%d-%H%M")
        
        ## PyTorch Loss Functions
        loss_fn = None
        
        # Define input and output chunk lengths
        input_chunk = 192   
        output_chunk = 64
        
        # Define the model
        my_model = TFTModel(
            input_chunk_length=input_chunk,             # Длина входного ля анализа, но это
            output_chunk_length=output_chunk,             # Длина выходногоо снижа
            hidden_size=160,                # Размер скрытых слоев в модели (предыдущее значение: 256) 
            hidden_continuous_size=64,
            lstm_layers=1,                  # Количество слоев LSTM в модели
            # feed_forward='GELU',                # [“GLU”, “Bilinear”, “ReGLU”, “GEGLU”, “SwiGLU”, “ReLU”, “GELU”]
            num_attention_heads=8,          # Количество голов внимания в механизме внимания (предыдущее значение: 8)
            full_attention=True,            # Флаг, использовать ли полный механизм внимания
            dropout=0.1,                    # Вероятность "выключения" нейронов для регуляризации модели
            batch_size=256,                  # Размер мини-выборки для одного шага обучения
            n_epochs=1,                     # Количество эпох обучения модели (предыдущее значение: 100)
            loss_fn=None,                   # Функция потерь, используемая для обучения модели
            optimizer_cls=torch.optim.AdamW,  # Изменен оптимизатор на AdamW
            optimizer_kwargs={
                'lr': 1e-3,                     # Скорость обучения
                'weight_decay': 1e-4,
                'fused': True,
                # 'amsgrad': True,
            },
            lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
            lr_scheduler_kwargs={
                'monitor': 'val_loss',
                'patience': 5,
                'factor': 0.5,
            },
            likelihood=QuantileRegression(quantiles=[0.01, 0.10, 0.30, 0.50, 0.70, 0.90, 0.99]),
            # likelihood=torch.nn.MSELoss(),
            add_encoders={
                'cyclic': {'future': ['day_of_year']},
                'datetime_attribute': {'future': ['is_month_start', 'is_month_end']},
            },
            pl_trainer_kwargs={
                "accelerator": 'gpu',
                "devices": 1,
                # "strategy": 'ddp',
                "precision": '32-true',
                "gradient_clip_val": 1,
                "callbacks": [early_stop_callback],
                "enable_checkpointing": False,
                "logger": False,
            },
            random_state=42,
            force_reset=False,
            save_checkpoints=False,
            log_tensorboard=False,
            model_name=script_name,
        )

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        
        # Set the number of epochs per evaluation
        n_epochs_per_eval = 1  # Frequency of metric calculation (every epoch)
        
        model_files = glob.glob(os.path.join(model_folder, "*.pt"))
        
        print_once(f"\n\033[1;32m  FITTING TRAINING SETTINGS: Total Epochs: {my_model.n_epochs} / Model: {(my_model.__class__.__name__)} \033[0m \n")
        print_once(f"\033[32m    - app: {script_name} / loss_fn: {(loss_fn.__class__.__name__ if loss_fn else 'None')} / / input_chunk: {input_chunk}, output_chunk: {output_chunk} \033[0m")
     
        
        model_files = glob.glob(os.path.join(model_folder, "*.pt"))
        if model_files:
            # AdamW=0
            latest_model_file = max(model_files, key=os.path.getmtime)
            # torch.serialization.add_safe_globals([TFTModel])
            # torch.serialization.add_safe_globals([QuantileRegression])
            # all_my_module_objs = []
            # for name in dir(my_model):
                # obj = getattr(my_model, name)
                # all_my_module_objs.append(obj)
            # torch.serialization.add_safe_globals(all_my_module_objs)
            my_model.load_weights(latest_model_file, map_location=torch.device('cuda'))
            print_once(f"\033[32m    - model loading: успешно загружен последний сохраненный файл: {latest_model_file}\033[0m")
        else:
            print_once(f"{line}\n")  # Adds an empty line
            print_once("\033[32m    - model loading: не найдено ни одного .pt файла в папке models.\033[0m")
        print_once(f"\n{line}\n")  # Adds an empty line
        time.sleep(3)
        # Train the model and calculate metrics after several epochs
        for epoch in range(0, my_model.n_epochs, n_epochs_per_eval): 
            
            trainer_fit = Trainer(
                accelerator="gpu",
                devices=-1,
                # strategy="ddp",
                # accumulate_grad_batches=7,
                precision='32-true',
                gradient_clip_val=1,
                callbacks=[early_stop_callback],
                max_epochs=my_model.n_epochs,
                enable_checkpointing=False,
                logger=False,
            )
            
            print_once("\n")
            print_once(f"\033[1;32m  TRAINING: Epochs in Session {epoch + n_epochs_per_eval} out of {my_model.n_epochs} for {(my_model.__class__.__name__)} \033[0m \n")
            
            start_time = time.time()

            # Train the model
            my_model.fit(
                series=train_transformed,
                past_covariates=train_cov_transformed,
                future_covariates=train_future_covariates_transformed,
                val_series=val_transformed,
                val_past_covariates=val_cov_transformed,
                val_future_covariates=val_future_covariates_transformed,
                epochs=n_epochs_per_eval,
                verbose=True,
                dataloader_kwargs=dataloader_kwargs,
                trainer=trainer_fit,
            )
            
            print_once(f"\n\n{line}\n")
            print_once(f"\033[1;32m STARTING FEATURE EVALUATION \033[0m ")
            
            trainer = my_model.trainer
            
            if trainer.num_devices < 1:
                print_once(" explainer не запускае так как у нас тут MultiGPUs")
            else:
                print_once(" explainer запускаем, все круто!\n\n")
                explainer = TFTExplainer(my_model)
                
                results_val = explainer.explain(
                    foreground_series=val_transformed,
                    foreground_past_covariates=val_cov_transformed,
                    foreground_future_covariates=future_covariates_full_transformed,
                )
                
                results_train = explainer.explain(
                    foreground_series=train_transformed,
                    foreground_past_covariates=train_cov_transformed,
                    foreground_future_covariates=future_covariates_full_transformed,
                )
                
                time.sleep(3)
                
                importances_train = results_train.get_feature_importances()
                importances_val = results_val.get_feature_importances()

                print_once(f"\n\n\033[1;32m importances_train \033[0m")
                print_sorted_importances(importances_train)

                print_once(f"\n\n\033[1;32m importances_val \033[0m")
                print_sorted_importances(importances_val)

                
                # показываем отбор фич
                explainer.plot_variable_selection(results_val, fig_size=(24, 45))
                plt.savefig(f"models/fival/{script_name}_{timestamp}_validation_variable_selection_plot.png")
                plt.close()

                
                explainer.plot_variable_selection(results_train, fig_size=(24, 45))
                plt.savefig(f"models/fitrain/{script_name}_{timestamp}_train_variable_selection_plot.png")
                plt.close()
                
                time.sleep(3)
                



                
                print_once(f"\n\n\033[1;32m runpodctl send models/fival/{script_name}_{timestamp}_validation_variable_selection_plot.png\033[0m")
                print_once(f"\033[1;32m runpodctl send models/fitrain/{script_name}_{timestamp}_train_variable_selection_plot.png\033[0m\n\n")
          
            print_once(f"\033[1;32m FEATURE EVALUATION FINISHED \033[0m")

            train_start = train_transformed.time_index.min()
            train_end = train_transformed.time_index.max()
            
            end_time = time.time()
            # Calculate elapsed time in minutes
            elapsed_time_minutes = (end_time - start_time) / 60

            predict_start_time = time.time()
            timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d / %H-%M-%S")
            print_once(f"{line}\n")
            
            if my_model.trainer.optimizers:
                current_lr = my_model.trainer.optimizers[0].param_groups[0]['lr']
                print_once(f"\033[32m learning rate on last Epoch: {current_lr}  \033[0m ")
            else:
                print_once(f"\033[32m learning rate on last Epoch: No optimizers found!  \033[0m ")
            
            print_once(f"\033[32m Elapsed time: {elapsed_time_minutes:.2f} minutes  \033[0m \n")
            print_once(f"\033[1;32m STARTING PREDICTION ON TRAINED MODEL - Epoch {epoch + n_epochs_per_eval} - Time: {timestamp_now} \033[0m \n")

            # Slice covariates for prediction
            input_chunk_length = my_model.input_chunk_length
            output_chunk_length = my_model.output_chunk_length
            
            # Calculate prediction horizon
            horizon = len(val_cov_transformed) + output_chunk_length

            print_once(f"  DEBUG: After Model.fit before eval: Checking prediction horizon parameter: {horizon}")

            # Calculate required start and end times for covariate data for prediction
            required_cov_start_time = val_transformed.start_time() - (input_chunk_length) * val_transformed.freq
            required_cov_end_time = val_transformed.end_time() + (output_chunk_length) * val_transformed.freq

            print_once(f"  DEBUG: After Model.fit before eval: Required start time for covariate data for prediction: {required_cov_start_time}")
            print_once(f"  DEBUG: After Model.fit before eval: Required end time for covariate data for prediction: {required_cov_end_time}")

            val_cov_predict = covariate_series_transformed.slice(required_cov_start_time, required_cov_end_time)
            
            print_once(f"{line}\n")

            # trainer_predict = Trainer(
                # inference_mode=True,
                # accelerator="gpu",
                # devices=1,
                # # strategy="ddp_spawn",
                # precision='32-true',
                # enable_progress_bar=False,
                # enable_checkpointing=False,
                # logger=False,
            # )

            # Generate predictions
            # pred_series = my_model.predict(
                # # trainer=trainer_predict,
                # n=horizon,
                # past_covariates=val_cov_predict,
                # future_covariates=future_covariates_full_transformed,
                # show_warnings=True,
                # num_samples=100,
                # verbose=True,
            # )
            
            # Предполагаем, что my_model, horizon, val_cov_predict и future_covariates_full_transformed уже определены
            # pred_series = run_prediction(my_model, horizon, val_cov_predict, future_covariates_full_transformed)
            pred_series = run_prediction(my_model, horizon, val_cov_predict, future_covariates_full_transformed)







            predict_end_time = time.time()
            predict_elapsed_time_minutes = (predict_end_time - predict_start_time) / 60

            pred_start_time = pred_series.start_time()
            pred_end_time = pred_series.end_time()
            start_time_zoomed = pred_end_time - timedelta(days=15)

            print_once(f"{line}\n")
            print_once(f"\033[1;32m  PREDICTION SCORING - Epoch {epoch + n_epochs_per_eval} - GENERATED IN: {predict_elapsed_time_minutes:.2f} MINUTES  \033[0m")
            print_once(f"  DEBUG: Prediction date range pred_series:", pred_series.time_index.min(), "to", pred_series.time_index.max())
            print_once(f"\n{line}\n")

            # Calculate metrics
            rmse_val = rmse(val_transformed, pred_series)
            mae_val = mae(val_transformed, pred_series)
            mape_val = mape(val_transformed, pred_series)
            r2_val = r2_score(val_transformed, pred_series)

            # Определяем диапазоны
            rmse_ranges = [0.1, 0.2]
            mae_ranges = [0.1, 0.2]
            mape_ranges = [5, 10]
            r2_ranges = [0.8, 0.6]

            # Получаем дельты через нашу новую функцию
            rmse_delta, mae_delta, mape_delta, r2_delta = track_metric_changes(rmse_val, mae_val, mape_val, r2_val)

            # Печатаем с дельтами (если есть старые значения)
            print_once(f"  - RMSE: {color_text(rmse_val, rmse_ranges)} {rmse_delta}")
            print_once(f"  - MAE: {color_text(mae_val, mae_ranges)} {mae_delta}")
            print_once(f"  - MAPE: {color_text(mape_val, mape_ranges)}% {mape_delta}")
            print_once(f"  - R2 Score: {color_text(r2_val, r2_ranges, is_r2=True)} {r2_delta}\n")
            print_once(
                f"  - Epoch Passed {epoch + n_epochs_per_eval} out of {my_model.n_epochs} "
                f"- RMSE: {color_text(rmse_val, rmse_ranges)} / "
                f"MAE: {color_text(mae_val, mae_ranges)} / "
                f"MAPE: {color_text(mape_val, mape_ranges)} / "
                f"R2 Score: {color_text(r2_val, r2_ranges, is_r2=True)}"
            )
            print_once("\n")

            timestamp2 = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            # Save the model
            if r2_val > -40:
                file_name = f"{timestamp2}-{(my_model.__class__.__name__)}_{('None' if loss_fn is None else loss_fn.__class__.__name__)}_TRAINING_{freq}-{input_chunk}-{output_chunk}_R2-{r2_val:.4f}.pt"
                my_model.save(f"models/{file_name}")
                print_once(f"\n\033[32m  DEBUG: Model saved as models/{file_name}\033[0m\n")
            else:
                print_once(f"\n\033[31m  DEBUG: Model not saved as R2 score is below threshold 0.70: {r2_val:.4f}\033[0m\n")

            print_once(f"{line}\n")
            
            
            time.sleep(9)

            # Inverse transform for plotting
            series_real = scaler_series.inverse_transform(val_transformed)
            pred_real = scaler_series.inverse_transform(pred_series)
            series_real_zoomed = series_real.slice(start_time_zoomed, pred_end_time)
            pred_real_zoomed = pred_real.slice(start_time_zoomed, pred_end_time)

            # Plot prediction and real data
            plt.figure(figsize=(250, 9))
            series_real.plot(label="Real Price", color='blue', linewidth=1, alpha=0.8)
            pred_real.plot(label="Predicted Median", color='red', linewidth=1)

            ax = plt.gca()
            ax.yaxis.set_major_locator(AutoLocator())
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=4))
            plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
            plt.grid(visible=True, which='major', color='grey', linestyle='-', linewidth=0.5)
            plt.grid(visible=True, which='minor', color='lightgrey', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=90)
            ax_right = ax.twinx()
            ax_right.set_ylim(ax.get_ylim())
            ax_right.yaxis.set_major_locator(AutoLocator())
            ax_right.set_ylabel("Values (duplicated axis)")

            plt.title(
                f"Model Validation {(my_model.__class__.__name__)} - {epoch + n_epochs_per_eval} - {freq} "
                f"- {train_start} to {train_end} - input: {input_chunk} / output: {output_chunk} "
                f"- R2: {r2_val:.4f} - RMSE: {rmse_val:.4f} - MAE: {mae_val:.4f} - MAPE: {mape_val:.4f}"
            )
            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.tight_layout()

            print_once("\n\033[1;32m  SAVING RESULTS\n\033[0m")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            plot_filename = f"{timestamp}-{(my_model.__class__.__name__)}_{('None' if loss_fn is None else loss_fn.__class__.__name__)}_FULLDATA_{freq}-{input_chunk}-{output_chunk}_R2-{r2_val:.4f}.png"
            plt.savefig(os.path.join(model_folder, plot_filename))
            plt.close()
            print_once(f"\033[32m  DEBUG: Saved plot to models/{plot_filename}\033[0m")

            # Zoomed plot
            plt.figure(figsize=(60, 9))
            
            lowest_q2, lowest_q, low_q, high_q, highest_q, highest_q2 = 0.01, 0.10, 0.30, 0.70, 0.90, 0.99

            label_q_outer2 = f"{int(lowest_q2 * 100)}-{int(highest_q2 * 100)}th percentiles"
            label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
            label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

            series_real_zoomed.plot(label="Real Price", color='black', linewidth=1, alpha=0.8)
            # pred_real_zoomed.plot(label="Predicted", color='Blue', linewidth=1, alpha=0.6)
            pred_real_zoomed.plot(low_quantile=lowest_q2, high_quantile=highest_q2, label=label_q_outer2, alpha=0.15)
            pred_real_zoomed.plot(low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer, alpha=0.3)
            pred_real_zoomed.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner, alpha=0.6)

            ax = plt.gca()
            ax.yaxis.set_major_locator(MultipleLocator(5))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=3))
            plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
            plt.grid(visible=True, which='major', color='grey', linestyle='-', linewidth=0.5)
            plt.grid(visible=True, which='minor', color='lightgrey', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=90)
            ax_right = ax.twinx()
            ax_right.set_ylim(ax.get_ylim())
            ax_right.yaxis.set_major_locator(MultipleLocator(5))
            ax_right.set_ylabel("Values (duplicated axis)")
            plt.title(
                f"Model Validation Zoomed {(my_model.__class__.__name__)} - {epoch + n_epochs_per_eval} - {freq}  "
                f"- {train_start} to {train_end} - input: {input_chunk} / output: {output_chunk} "
                f"- R2 {r2_val:.4f} - RMSE {rmse_val:.4f} - MAE {mae_val:.4f} - MAPE {mape_val:.4f}"
            )
            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.tight_layout()

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            plot_filename = f"{timestamp}-{(my_model.__class__.__name__)}_{('None' if loss_fn is None else loss_fn.__class__.__name__)}_PREDREAL-ZOOMED_{freq}-{input_chunk}-{output_chunk}_R2-{r2_val:.4f}.png"
            plt.savefig(os.path.join(model_folder, plot_filename))
            plt.close()
            print_once(f"\033[32m  DEBUG: Saved plot to models/{plot_filename}\033[0m")

            if r2_val > -40:
                file_name = f"{timestamp}-{(my_model.__class__.__name__)}_{('None' if loss_fn is None else loss_fn.__class__.__name__)}_TRAINING_{freq}-{input_chunk}-{output_chunk}_R2-{r2_val:.4f}.pt"
                my_model.save(f"models/{file_name}")
                print_once(f"\n\033[32m  DEBUG: Model saved as models/{file_name}\033[0m\n")
            else:
                print_once(f"\n\033[31m  DEBUG: Model not saved as R2 score is below threshold 0.70: {r2_val:.4f}\033[0m\n")

            print_once(f"{line}\n")
            
            
            time.sleep(9)


            
        def eval_model(my_model, val_transformed, val_cov_transformed, model_folder, horizon, freq):
            """
            Evaluates the model on the validation set and saves results.
            """
            print_once("\033[1;31m  EVAL FUNCTION TURNED OFF \033[0m  \n")
            print_once(f"{line}\n")

        eval_model(my_model, val_transformed, val_cov_transformed, model_folder, horizon, freq)

    except Exception as e:
        tb = traceback.format_exc()
        print_once(f"\033[1;31m  Error processing: {str(e)}\033[0m")
        print_once(f"\033[1;31m  Traceback details:\033[0m")
        print_once(tb)

def main():
    """
    Main function to initialize directories, set device configurations,
    and process each CSV file in the data directory.
    """
    # Ensure the necessary directories exist
    clean_data_folder = './'
    model_folder = 'models'
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs('models/fitrain', exist_ok=True)
    os.makedirs('models/fival', exist_ok=True)

    # Determine terminal width for formatting
    terminal_width = shutil.get_terminal_size().columns
    line = '-' * terminal_width
        
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32  # Using float32 instead of bfloat16

    print_once("\nInitialization")
    print_once(f"{line}\n")
    print_once(f"Device: {device}")
    print_once(f"Is CUDA available: {torch.cuda.is_available()}")
    print_once(f"Selected dtype: {dtype}")
    
    script_name = os.path.basename(__file__)
    print_once(f"Имя скрипта: {script_name}")
    
    # Process each CSV file in the clean data folder
    for file_name in os.listdir(clean_data_folder):
        if file_name.endswith('.csv'):
            print_once(f"DEBUG: Processing file: {file_name}")
            process_file(file_name, clean_data_folder, model_folder)

if __name__ == '__main__':
    main()
