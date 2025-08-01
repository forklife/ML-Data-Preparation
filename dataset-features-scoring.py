#!/usr/bin/env python3
# feature_evaluation.py
# Version: 1.9 (только LightGBM на GPU с SHAP для interactions)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
import traceback
from lightgbm import LGBMRegressor
from datetime import datetime
from colorama import Fore, Style, init
import tempfile
import shutil
from sklearn.preprocessing import StandardScaler
import shap  # Для SHAP values с interactions

# Инициализация colorama и логирования (без изменений)
init(autoreset=True)
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.WARNING:
            record.msg = Fore.RED + record.msg + Style.RESET_ALL
        elif record.levelno == logging.INFO:
            record.msg = Fore.GREEN + record.msg + Style.RESET_ALL
        else:
            record.msg = Fore.YELLOW + record.msg + Style.RESET_ALL
        return super().format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler("feature_evaluation.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def create_output_folder():
    now = datetime.now()
    folder_name = os.path.join("feature-scoring", now.strftime('%m-%d-%Y_%H-%M'))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name

def get_filename_with_time(model_name, prefix, file_type="csv"):
    now = datetime.now().strftime("%d-%m-%Y_%H-%M")
    return f"{prefix}_{model_name}_{now}.{file_type}"

def compute_log_mean(values):
    positive_vals = values[values > 0]
    if not positive_vals.empty:
        return np.exp(np.mean(np.log(positive_vals)))
    else:
        return values.mean()

def plot_feature_importance(importances, title, output_path):
    if importances.empty:
        logger.error(f"Cannot plot feature importances for {title}: empty data.")
        return
    plt.figure(figsize=(18, 8))
    importances_sorted = importances.sort_values(ascending=False)
    importances_sorted.plot(kind='bar')
    log_mean = compute_log_mean(importances_sorted)
    plt.axhline(y=log_mean, color='red', linestyle='--', label='Log Mean')
    plt.legend()
    plt.yscale('log')
    plt.title(title)
    plt.ylabel('Importance (log scale)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved feature importance plot: {output_path}")

def select_features_lightgbm(X, y, n_estimators=500, n_jobs=-1):
    lgb = LGBMRegressor(n_estimators=n_estimators, random_state=42, n_jobs=n_jobs, verbose=1)  # GPU enabled
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    lgb.fit(X_scaled, y)
    
    # SHAP для importance с interactions
    explainer = shap.TreeExplainer(lgb)
    shap_values = explainer.shap_values(X_scaled)
    importances = np.abs(shap_values).mean(axis=0)  # Mean abs SHAP - учитывает interactions
    feature_importance_lgb = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    selected_features_lgb = feature_importance_lgb[feature_importance_lgb > 0].index.tolist()
    return selected_features_lgb, feature_importance_lgb

def save_features_and_scores(selected_features, feature_importance, model_name, file_path, output_folder):
    output_file = os.path.join(output_folder, get_filename_with_time(model_name, os.path.splitext(os.path.basename(file_path))[0]))
    df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance[selected_features].values})
    df.to_csv(output_file, index=False)
    logger.info(f"Saved feature scores for {model_name}: {output_file}")

def process_file(file_path, output_folder):
    file_results = {}
    try:
        logger.info(f"Processing file: {file_path}")
        df = pd.read_csv(file_path, encoding='utf-8')
        df.columns = df.columns.str.strip().str.lower()
        required_columns = ['datetime', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"File {file_path} missing required columns: {missing_columns}")
            return {}
        
        df = df.dropna(subset=['close', 'datetime'])
        excluded_columns = ['id', 'datetime', 'close']
        feature_columns = [col for col in df.columns if col not in excluded_columns]
        logger.info(f"Using {len(feature_columns)} features: {feature_columns}")
        X = df[feature_columns]
        y = df['close']

        model_name = "LightGBM_SHAP"
        selected_features, feature_importance = select_features_lightgbm(X, y)
        save_features_and_scores(selected_features, feature_importance, model_name, file_path, output_folder)
        plot_feature_importance(
            feature_importance,
            f"{os.path.basename(file_path)} - Feature Importance {model_name}",
            os.path.join(output_folder, get_filename_with_time(f"feature_importance_{model_name.lower().replace(' ', '_')}", os.path.splitext(os.path.basename(file_path))[0], "png"))
        )
        file_results[model_name] = feature_importance[feature_importance > 0].to_dict()
        return file_results
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        logger.error(traceback.format_exc())
        return {}

def main():
    logger.info("Program started.")
    temp_dir = tempfile.mkdtemp(dir=".")
    tempfile.tempdir = os.path.abspath(temp_dir)
    os.environ["TMPDIR"] = os.path.abspath(temp_dir)
    logger.info(f"Created temporary folder at: {os.environ['TMPDIR']}")
    
    try:
        input_folder = 'output-datasets-fortraining'
        if not os.path.exists(input_folder):
            logger.error(f"Folder {input_folder} does not exist.")
            return
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv') and 'features' not in f and 'importance' not in f]
        if not csv_files:
            logger.error(f"No CSV files found in {input_folder}.")
            return
        logger.info(f"Found {len(csv_files)} CSV files for processing.")
        output_folder = create_output_folder()
        
        methods = ["LightGBM_SHAP"]
        aggregated_results = {method: {} for method in methods}
        
        all_X = pd.DataFrame()  # Для корреляции, если нужно
        
        for file in csv_files:
            file_path = os.path.join(input_folder, file)
            file_results = process_file(file_path, output_folder)
            for model_name, feat_dict in file_results.items():
                for feature, imp in feat_dict.items():
                    aggregated_results[model_name][feature] = aggregated_results[model_name].get(feature, 0) + imp
            
            # Собираем данные для корреляции (опционально)
            df = pd.read_csv(file_path, encoding='utf-8')
            df.columns = df.columns.str.strip().str.lower()
            df = df.dropna(subset=['close', 'datetime'])
            excluded_columns = ['id', 'datetime', 'close']
            feature_columns = [col for col in df.columns if col not in excluded_columns]
            X_file = df[feature_columns]
            all_X = pd.concat([all_X, X_file], ignore_index=True)
        
        union_results = {}
        for model_name, feat_dict in aggregated_results.items():
            for feature, imp in feat_dict.items():
                if feature not in union_results:
                    union_results[feature] = {"total_importance": imp, "count": 1}  # Поскольку один метод
                else:
                    union_results[feature]["total_importance"] += imp
                    union_results[feature]["count"] += 1

        # Поскольку один метод, common = все
        top_common = sorted(union_results.items(), key=lambda x: x[1]["total_importance"], reverse=True)
        df_common = pd.DataFrame({"Feature": [f for f, data in top_common], "Total Importance": [data["total_importance"] for f, data in top_common]})
        df_common["Group"] = "Common"
        df_non_common = pd.DataFrame()  # Нет non-common
        final_df = df_common.copy()

        final_csv_path = os.path.join(output_folder, "final.csv")
        final_df.to_csv(final_csv_path, index=False)
        logger.info(f"Saved final CSV: {final_csv_path}")

        # Финальный график (только common, так как один метод)
        fig, ax = plt.subplots(figsize=(27, 8))
        if not df_common.empty:
            ax.bar(df_common["Feature"], df_common["Total Importance"])
            geo_mean_common = compute_log_mean(df_common["Total Importance"])
            ax.axhline(y=geo_mean_common, color='red', linestyle='--', label='Log Mean')
            ax.set_title("All Features (LightGBM with SHAP)")
            ax.set_ylabel("Total Importance (SHAP)")
            ax.tick_params(axis='x', rotation=90)
            ax.legend()
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        final_graph_path = os.path.join(output_folder, "final_plot.png")
        plt.savefig(final_graph_path)
        plt.close()
        logger.info(f"Saved final plot: {final_graph_path}")

        logger.info("Processing completed.")
        logger.info("Program ended.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Removed temporary folder: {temp_dir}")

if __name__ == "__main__":
    main()