#!/usr/bin/env python3
# improved_feature_evaluation.py
# Version: 2.3 (Updated final plot width and title with feature count)

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
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.ensemble import RandomForestRegressor  # For simulated annealing objective

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
    plt.figure(figsize=(12, 18))  # Adapted size: wider for labels, taller for many features
    importances_sorted = importances.sort_values(ascending=True)  # For horizontal bar
    importances_sorted.plot(kind='barh')  # Horizontal bar plot, features on Y-axis
    log_mean = compute_log_mean(importances_sorted)
    plt.axvline(x=log_mean, color='red', linestyle='--', label='Log Mean')  # Vertical line now
    plt.legend()
    plt.xscale('log')  # Log scale on X (importance)
    plt.title(title)
    plt.xlabel('Importance (log scale)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved feature importance plot: {output_path}")

def check_stationarity(series, pvalue_threshold=0.05):
    """Perform ADF test and return if stationary, and differenced series if not."""
    cleaned = series.dropna()
    if len(cleaned) < 2 or cleaned.std() == 0:
        logger.info(f"Constant or too short series: {series.name}, treating as stationary.")
        return True, series
    try:
        result = adfuller(cleaned)
        if result[1] > pvalue_threshold:  # Non-stationary
            return False, series.diff()  # Keeps index, NaN in first
        return True, series
    except ValueError as e:
        if "constant" in str(e).lower():
            logger.info(f"Caught constant error in adfuller for {series.name}, treating as stationary.")
            return True, series
        raise

def handle_stationarity(X):
    """Check and difference non-stationary features."""
    for col in X.columns:
        is_stat, new_series = check_stationarity(X[col])
        if not is_stat:
            logger.info(f"Differencing non-stationary feature: {col}")
            X[col] = new_series
    X.dropna(inplace=True)  # Drop rows with NaN (e.g., first row if any differenced)
    return X

def compute_vif(X, vif_threshold=10):
    """Compute VIF and remove features with high multicollinearity."""
    X = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    # Handle inf/nan VIF as high
    vif_data['VIF'] = vif_data['VIF'].replace([np.inf, -np.inf], np.nan)
    high_vif = vif_data[(vif_data['VIF'] > vif_threshold) | vif_data['VIF'].isna()]['feature'].tolist()
    if 'const' in high_vif:
        high_vif.remove('const')
    logger.info(f"Removing high VIF features: {high_vif}")
    return X.drop(columns=high_vif).drop(columns=['const'], errors='ignore'), high_vif

def select_features_granger(X, y, max_lag=5, pvalue_threshold=0.05):
    """Use Granger causality to select causal features."""
    causal_features = []
    for col in X.columns:
        try:
            data = pd.concat([y, X[col]], axis=1).dropna()
            result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            pvalues = [result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
            if min(pvalues) < pvalue_threshold:
                causal_features.append(col)
        except:
            pass
    logger.info(f"Granger causal features: {causal_features}")
    return causal_features

def train_model_for_annealing(X, y, feature_subset, cv=3):
    """Objective function for simulated annealing: CV MSE with RandomForest."""
    if len(feature_subset) == 0:
        return np.inf  # Invalid subset
    X_subset = X.iloc[:, feature_subset]
    model = RandomForestRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = []
    for train_idx, val_idx in tscv.split(X_subset):
        X_train, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        scores.append(np.mean((model.predict(X_val) - y_val)**2))  # MSE
    return np.mean(scores)  # Lower is better

def simulated_annealing(X, y, initial_temp=1.0, cooling_rate=0.95, max_iterations=100, min_features=1, max_features=None):
    """Simulated annealing for optimal feature subset selection."""
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features
    
    # Initialize with random subset
    current_features = np.random.choice(range(n_features), size=np.random.randint(min_features, max_features+1), replace=False)
    current_score = train_model_for_annealing(X, y, current_features)
    best_features = current_features.copy()
    best_score = current_score
    
    temperature = initial_temp
    for i in range(max_iterations):
        # Perturb: add/remove/replace a feature
        new_features = list(current_features.copy())
        action = np.random.choice(['add', 'remove', 'replace'])
        if action == 'add' and len(new_features) < max_features:
            available = list(set(range(n_features)) - set(new_features))
            if available:
                new_features.append(np.random.choice(available))
        elif action == 'remove' and len(new_features) > min_features:
            remove_idx = np.random.randint(len(new_features))
            new_features.pop(remove_idx)
        elif action == 'replace' and len(new_features) > 0:
            remove_idx = np.random.randint(len(new_features))
            available = list(set(range(n_features)) - set(new_features))
            if available:
                new_features[remove_idx] = np.random.choice(available)
        
        new_features = sorted(new_features)  # For consistency
        new_score = train_model_for_annealing(X, y, new_features)
        
        # Acceptance
        if new_score < current_score:  # Better (lower MSE)
            current_features = new_features
            current_score = new_score
            if new_score < best_score:
                best_features = new_features
                best_score = new_score
        else:
            delta = new_score - current_score
            acceptance_prob = np.exp(-delta / temperature)
            if np.random.rand() < acceptance_prob:
                current_features = new_features
                current_score = new_score
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.01:
            break
    
    logger.info(f"Best annealed features: {best_features}, Score: {best_score}")
    return [X.columns[i] for i in best_features]

def select_features_lightgbm(X, y, n_splits=5, n_estimators=500, n_jobs=-1, percentile=90):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    shap_importances = np.zeros(X.shape[1])
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        
        lgb = LGBMRegressor(n_estimators=n_estimators, random_state=42, n_jobs=n_jobs, verbose=-1)
        lgb.fit(X_train_scaled, y_train)
        
        explainer = shap.TreeExplainer(lgb)
        shap_values = explainer.shap_values(X_val_scaled)
        shap_importances += np.abs(shap_values).mean(axis=0)
    
    shap_importances /= n_splits  # Average over folds
    feature_importance_lgb = pd.Series(shap_importances, index=X.columns).sort_values(ascending=False)
    
    # Threshold tuning: top percentile
    threshold = np.percentile(feature_importance_lgb[feature_importance_lgb > 0], percentile)
    selected_features_lgb = feature_importance_lgb[feature_importance_lgb >= threshold].index.tolist()
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
        df = df.sort_values('datetime')  # Ensure time order
        excluded_columns = ['id', 'datetime', 'close']
        feature_columns = [col for col in df.columns if col not in excluded_columns]
        logger.info(f"Using {len(feature_columns)} features: {feature_columns}")
        X = df[feature_columns].copy()  # Copy to avoid SettingWithCopyWarning
        y = df['close']
        
        # New: Handle stationarity
        X = handle_stationarity(X)
        y = y.loc[X.index]  # Align y with X after drops
        
        # New: Remove constant features before VIF
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            logger.info(f"Removing constant features: {constant_cols}")
            X.drop(columns=constant_cols, inplace=True)
        
        # New: Remove multicollinear features with VIF
        X, removed_vif = compute_vif(X)
        
        # New: Granger causality selection
        granger_features = select_features_granger(X, y)
        X_granger = X[granger_features]
        
        # Combine with LightGBM SHAP (using CV)
        model_name = "LightGBM_SHAP_CV"
        selected_features, feature_importance = select_features_lightgbm(X, y)
        save_features_and_scores(selected_features, feature_importance, model_name, file_path, output_folder)
        plot_feature_importance(
            feature_importance,
            f"{os.path.basename(file_path)} - Feature Importance {model_name}",
            os.path.join(output_folder, get_filename_with_time(f"feature_importance_{model_name.lower().replace(' ', '_')}", os.path.splitext(os.path.basename(file_path))[0], "png"))
        )
        file_results[model_name] = feature_importance[feature_importance > 0].to_dict()
        
        # New: Simulated annealing on Granger pre-selected features
        if not X_granger.empty:
            annealed_features = simulated_annealing(X_granger, y)
            file_results["Annealed"] = {feat: 1.0 for feat in annealed_features}  # Dummy importance
        
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
        
        methods = ["LightGBM_SHAP_CV", "Annealed"]
        aggregated_results = {method: {} for method in methods}
        
        for file in csv_files:
            file_path = os.path.join(input_folder, file)
            file_results = process_file(file_path, output_folder)
            for model_name, feat_dict in file_results.items():
                for feature, imp in feat_dict.items():
                    aggregated_results[model_name][feature] = aggregated_results[model_name].get(feature, 0) + imp
            
        # Aggregation and final saves (similar to original, adapted for new methods)
        union_results = {}
        for model_name, feat_dict in aggregated_results.items():
            for feature, imp in feat_dict.items():
                if feature not in union_results:
                    union_results[feature] = {"total_importance": imp, "count": 1}
                else:
                    union_results[feature]["total_importance"] += imp
                    union_results[feature]["count"] += 1

        top_common = sorted(union_results.items(), key=lambda x: x[1]["total_importance"], reverse=True)
        df_common = pd.DataFrame({"Feature": [f for f, data in top_common], "Total Importance": [data["total_importance"] for f, data in top_common]})
        df_common["Group"] = "Common"
        final_df = df_common.copy()

        final_csv_path = os.path.join(output_folder, "final.csv")
        final_df.to_csv(final_csv_path, index=False)
        logger.info(f"Saved final CSV: {final_csv_path}")

        num_features = len(df_common)
        fig, ax = plt.subplots(figsize=(24, 27))  # Increased width by 3 times (from 8 to 24)
        if not df_common.empty:
            df_common_sorted = df_common.sort_values("Total Importance", ascending=True)
            ax.barh(df_common_sorted["Feature"], df_common_sorted["Total Importance"])
            geo_mean_common = compute_log_mean(df_common_sorted["Total Importance"])
            ax.axvline(x=geo_mean_common, color='red', linestyle='--', label='Log Mean')
            ax.set_title(f"All Features (Improved Methods) - {num_features} features")
            ax.set_xlabel("Total Importance")
            ax.tick_params(axis='y', rotation=0)
            ax.legend()
            ax.set_xscale('log')
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