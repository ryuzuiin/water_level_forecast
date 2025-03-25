import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime


class DynamicFeatureLibrary:
    def __init__(self, processed_dir: str, features_config_file: str = None):
        self.processed_dir = Path(processed_dir)
        self.ml_input_dir = self.processed_dir / "ml_input"
        self.ml_featured_dir = self.processed_dir / "ml_featured"

        os.makedirs(self.ml_featured_dir,exist_ok=True)

        self.setup_logging()
        self.logger.info("Initializing DynamicFeatureLibrary")
        
        self.features_config = self._load_features_config(features_config_file)
        self.logger.info(f"Features configuration loaded with {len(self.features_config)} headworks")

    def setup_logging(self):
        log_dir = self.processed_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"feature_engineering_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def zenpukuseki(h, B, W):
        if W >= h:
            return 0
        else:
            h = h - W
        e = 0 if W <= 1 else 0.55 * (W - 1)
        return (1.785 + (0.00295 / h + 0.237 * h / W) * (1 + e)) * B * (h ** (3 / 2))


    def _load_features_config(self, config_file):
        if config_file and os.path.exists(config_file):
            self.logger.info(f"Loading features configuration from file: {config_file}")
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
            
        else:
            self.logger.info("No configuration file found, creating default configuration")
            headworks = ["別所頭首工", "蒲生頭首工", "蓮花寺頭首工", "名神日野川頭首工"]

            base_features = ["Hour", "Weekday", "Month"]

            rainfall_features = [
                "第１段揚水機場 累計雨量", "管網系１０号分水工 累計雨量", "第２段揚水機場 累計雨量",
                "第３段揚水機場 累計雨量", "第４段揚水機場 累計雨量", "山本新田支線揚水機場 累計雨量",
                "蔵王ダム 累計雨量"
            ]

            headwork_specific = {
                "別所頭首工": [
                    "日野川ダム 全放流量", "小井口頭首工 右岸１取水量",  "小井口頭首工 右岸２取水量", "別所頭首工 河川水位", 
                    "別所頭首工 右岸取水量", "別所頭首工 越流流量", "別所頭首工 魚道流量", "別所頭首工 取水量",
                    "小井口頭首工 取水量"
                ],
                "蒲生頭首工": [
                    "日野川ダム 全放流量", "小井口頭首工 河川水位", "小井口頭首工 右岸１取水量",  "小井口頭首工 右岸２取水量", 
                    "別所頭首工 河川水位", "別所頭首工 右岸取水量", "必佐頭首工 河川水位",
                    "必佐頭首工 右岸取水量", "蒲生頭首工 河川水位", "蒲生頭首工 左岸取水量"
                ],
                "蓮花寺頭首工": [
                    "原頭首工 河川水位", "原頭首工 左岸取水量", "鳥居平頭首工 河川水位",
                    "鳥居平頭首工 左岸取水量", "鳥居平頭首工 右岸取水量", "蓮花寺頭首工 河川水位", "蓮花寺頭首工 右岸取水量", 
                    "蓮花寺頭首工 左岸取水量", "蓮花寺頭首工 取水量", "鳥居平頭首工 取水量"
                ],
                "名神日野川頭首工": [
                    "日野川ダム 全放流量", "小井口頭首工 河川水位", "小井口頭首工 右岸１取水量",  "小井口頭首工 右岸２取水量", 
                    "別所頭首工 河川水位", "別所頭首工 右岸取水量", "必佐頭首工 河川水位",
                    "必佐頭首工 右岸取水量", "蒲生頭首工 河川水位", "蒲生頭首工 左岸取水量",
                    "原頭首工 河川水位", "原頭首工 左岸取水量", "鳥居平頭首工 河川水位", "鳥居平頭首工 左岸取水量", 
                    "鳥居平頭首工 右岸取水量", "蓮花寺頭首工 河川水位", "蓮花寺頭首工 右岸取水量", "蓮花寺頭首工 左岸取水量", 
                    "名神日野川頭首工 河川水位", "名神日野川頭首工 左岸取水量",
                    "別所頭首工 越流流量", "別所頭首工 魚道流量", "別所頭首工 取水量", "小井口頭首工 取水量",
                    "蓮花寺頭首工 取水量", "鳥居平頭首工 取水量"
                ]
            }

            lag_windows = {
                10: list(range(10, 61, 10)),
                20: list(range(20, 71, 10)),
                30: list(range(30, 81, 10)),
                40: list(range(40, 91, 10)),
                50: list(range(50, 101, 10)),
                60: list(range(60, 111, 10)),
                120: list(range(120, 171, 10))
            }

            model_feature_combos = {
                 "10min_rate": {
                    "base_features": True,
                    "rainfall_features": True,
                    "specific_features": True,
                    "lag_times": [10, 20, 30, 40, 50, 60],
                    "target": "変動率",
                    "prediction_horizon": 10,
                    "include_target_lags": True,
                    "include_abs_lags": True
                    
                },
                "20min_rate": {
                    "base_features": True,
                    "rainfall_features": True,
                    "specific_features": True,
                    "lag_times": [10, 20, 30, 40, 50, 60],
                    "target": "変動率",
                    "prediction_horizon": 20,
                    "include_target_lags": True,
                    "include_abs_lags": True
                },
                "30min_rate": {
                    "base_features": True,
                    "rainfall_features": True,
                    "specific_features": True, 
                    "lag_times": [10, 20, 30, 40, 50, 60],
                    "target": "変動率",
                    "prediction_horizon": 30,
                    "include_target_lags": True,
                    "include_abs_lags": True
                },
                "40min_rate": {
                    "base_features": True,
                    "rainfall_features": True,
                    "specific_features": True,
                    "lag_times": [10, 20, 30, 40, 50, 60],
                    "target": "変動率",
                    "prediction_horizon": 40,
                    "include_target_lags": True,
                    "include_abs_lags": True
                },
                "50min_rate": {
                    "base_features": True,
                    "rainfall_features": True,
                    "specific_features": True,
                    "lag_times": [10, 20, 30, 40, 50, 60],
                    "target": "変動率",
                    "prediction_horizon": 50,
                    "include_target_lags": True,
                    "include_abs_lags": True
                },
                "60min_rate": {
                    "base_features": True,
                    "rainfall_features": True,
                    "specific_features": True,
                    "lag_times": [10, 20, 30, 40, 50, 60],
                    "target": "変動率",
                    "prediction_horizon": 60,
                    "include_target_lags": True,
                    "include_abs_lags": True
                },
                "120min_rate": {
                    "base_features": True,
                    "rainfall_features": True,
                    "specific_features": True,
                    "lag_times": [10, 20, 30, 40, 50, 60],
                    "target": "変動率",
                    "prediction_horizon": 120,
                    "include_target_lags": True,
                    "include_abs_lags": True
                },

            }

            config = {headwork: {} for headwork in headworks}

            for headwork in headworks:
                config[headwork]["base_features"] = base_features
                config[headwork]["rainfall_features"] = rainfall_features
                config[headwork]["specific_features"] = headwork_specific[headwork]
                config[headwork]["lag_times"] = [10, 20, 30, 40, 50, 60]
                config[headwork]["prediction_horizons"] = [10, 20, 30, 40, 50, 60, 120]
                config[headwork]["lag_windows"] = lag_windows
                config[headwork]["model_feature_combos"] = model_feature_combos

            self.logger.info(f"Created default configuration with {len(headworks)} headworks")
            return config
    
    def save_config(self, output_file):
        self.logger.info(f"Saving configuration to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.features_config, f, ensure_ascii=False, indent=2)
        self.logger.info("Configuration saved successfully")

    def get_features_for_headwork(self, headwork_name):
        self.logger.info(f"Getting features for headwork: {headwork_name}")
        if headwork_name not in self.features_config:
            error_msg = f"Headwork '{headwork_name}' is not in the configuration."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        config = self.features_config[headwork_name]
        all_features = (
            config["base_features"] + 
            config["rainfall_features"] + 
            config["specific_features"]
        )

        lag_features = []
        for feature in config["rainfall_features"] + config["specific_features"]:
            for lag in config["lag_times"]:
                lag_features.append(f"{feature}_{lag}lag")

        rate_lag_features = []
        abs_lag_features = []
        for horizon in config["prediction_horizons"]:
            for lag in config["lag_windows"][horizon]:
                rate_lag_features.append(f"{horizon}分後変動率_{headwork_name}_河川水位_{lag}lag")
                abs_lag_features.append(f"{horizon}分後変動_{headwork_name}_河川水位_{lag}lag")

        all_features_combined = all_features + lag_features + rate_lag_features + abs_lag_features
        self.logger.info(f"Generated {len(all_features_combined)} features for {headwork_name}")
        return all_features_combined
    

    def get_model_features(self, headwork_name, model_name):
        self.logger.info(f"Getting model features for headwork: {headwork_name}, model: {model_name}")
        if headwork_name not in self.features_config:
            error_msg = f"Headwork '{headwork_name}' is not in the configuration."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        config = self.features_config[headwork_name]
        if "model_feature_combos" not in config or model_name not in config["model_feature_combos"]:
            error_msg = f"Model '{model_name}' features are not defined."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        model_config = config["model_feature_combos"][model_name]
        selected_features = set()

        if model_config.get("base_features", False):
            selected_features.update(config["base_features"])
            
        if model_config.get("rainfall_features", False):
            selected_features.update(config["rainfall_features"])

        if model_config.get("specific_features", False):
            selected_features.update(config["specific_features"])

        lag_times = model_config.get("lag_times", [])
        for feature in (config["rainfall_features"] + config["specific_features"]):
            for lag in lag_times:
                lag_feature = f"{feature}_{lag}lag"
                selected_features.add(lag_feature)

        prediction_horizon = model_config.get("prediction_horizon", 60)
        lags = config["lag_windows"].get(prediction_horizon, [])
        for lag in lags:
            selected_features.add(f"{prediction_horizon}分後変動率_{headwork_name}_河川水位_{lag}lag")
            selected_features.add(f"{prediction_horizon}分後変動_{headwork_name}_河川水位_{lag}lag")

        for horizon in config["prediction_horizons"]:
            for lag in config["lag_windows"].get(horizon, []):
                selected_features.add(f"{horizon}分後変動率_{headwork_name}_河川水位_{lag}lag")
                selected_features.add(f"{horizon}分後変動_{headwork_name}_河川水位_{lag}lag")


        feature_list = list(selected_features)
        self.logger.info(f"Selected {len(feature_list)} features for model {model_name}")
        return feature_list
    
    
    def get_model_target(self, headwork_name, model_name):
        self.logger.info(f"Getting target for headwork: {headwork_name}, model: {model_name}")
        if headwork_name not in self.features_config:
            error_msg = f"Headwork '{headwork_name}' is not in the configuration."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        config = self.features_config[headwork_name]
        if "model_feature_combos" not in config or model_name not in config["model_feature_combos"]:
            error_msg = f"Model '{model_name}' features are not defined."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        
        model_config = config["model_feature_combos"][model_name]
        target_type = model_config.get("target", "変動率")
        horizon = model_config.get("prediction_horizon", 60)

        if target_type == "変動率":
            target = f"{horizon}分後変動率_{headwork_name}_河川水位"
        else:
            target = f"{horizon}分後変動_{headwork_name}_河川水位"
            
        self.logger.info(f"Target for model {model_name}: {target}")
        return target
        

    def prepare_headwork_data(self, headwork_name, input_file=None):
        if input_file is None:
            input_file = self.ml_input_dir / "with_rainfall_train.csv"
            self.logger.info(f"Using default input file for {headwork_name}: {input_file}")
        else:
            self.logger.info(f"Preparing data for {headwork_name} using file: {input_file}")

        if not isinstance(input_file, Path):
            input_file = Path(input_file)

        if not input_file.exists():
            error_msg = f"Data File does not exist: {input_file}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8-sig', parse_dates=["datetime"])
        self.logger.info(f"Loaded {len(df)} rows from {input_file}")

        # Calculate derived features using hydraulic formulas
        self.logger.info("Calculating derived hydraulic features")
        if "別所頭首工 河川水位" in df.columns:
            self.logger.info("Computing overflow discharge for 別所頭首工")
            df["別所頭首工 越流流量"] = df["別所頭首工 河川水位"].apply(
                lambda h: self.zenpukuseki(h - 0.5, 29.35, 2.1) + self.zenpukuseki(h, 10.85, 2.6)
            )
            df["別所頭首工 魚道流量"] = df["別所頭首工 河川水位"].apply(
                lambda h: self.zenpukuseki(h - 1.6, 0.6, 0.8) + self.zenpukuseki(h - 1.6, 1.4, 0.9)
            )

        if "小井口頭首工 右岸１取水量" in df.columns and "小井口頭首工 右岸２取水量" in df.columns:
            self.logger.info("Computing total intake for 小井口頭首工")
            df["小井口頭首工 取水量"] = df["小井口頭首工 右岸１取水量"] + df["小井口頭首工 右岸２取水量"]

        if {"別所頭首工 右岸取水量", "別所頭首工 越流流量", "別所頭首工 魚道流量"}.issubset(df.columns):
            self.logger.info("Computing total intake for 別所頭首工")
            df["別所頭首工 取水量"] = df["別所頭首工 右岸取水量"] + df["別所頭首工 越流流量"] + df["別所頭首工 魚道流量"]

        if {"蓮花寺頭首工 右岸取水量", "蓮花寺頭首工 左岸取水量"}.issubset(df.columns):
            self.logger.info("Computing total intake for 蓮花寺頭首工")
            df["蓮花寺頭首工 取水量"] = df["蓮花寺頭首工 右岸取水量"] + df["蓮花寺頭首工 左岸取水量"]

        if {"鳥居平頭首工 右岸取水量", "鳥居平頭首工 左岸取水量"}.issubset(df.columns):
            self.logger.info("Computing total intake for 鳥居平頭首工")
            df["鳥居平頭首工 取水量"] = df["鳥居平頭首工 右岸取水量"] + df["鳥居平頭首工 左岸取水量"]

        self.logger.info("Adding time-based features")
        if 'Hour' not in df.columns:
            df['Hour'] = df['datetime'].dt.hour
        if 'Weekday' not in df.columns:
            df['Weekday'] = df['datetime'].dt.dayofweek
        if 'Month' not in df.columns:
            df['Month'] = df['datetime'].dt.month

        self.logger.info(f"Generating lag features for {headwork_name}")
        for feature in self.features_config[headwork_name]["rainfall_features"] + self.features_config[headwork_name]["specific_features"]:
            if feature in df.columns:
                for lag in self.features_config[headwork_name]["lag_times"]:
                    lag_col = f"{feature}_{lag}lag"
                    
                    lag_df = df[["datetime", feature]].copy()
                    lag_df["datetime"] += pd.Timedelta(minutes=lag)

                    df = df.merge(lag_df.rename(columns={feature: lag_col}), on="datetime", how="left")

        water_level_col = f"{headwork_name} 河川水位"
        self.logger.info(f"Generating target variables using water level column: {water_level_col}")
        
        all_rate_columns = []
        all_abs_columns = []

        if water_level_col in df.columns:
            for horizon in self.features_config[headwork_name]["prediction_horizons"]:
                target_col = f"{horizon}分後変動率_{headwork_name}_河川水位"
                abs_col = f"{horizon}分後変動_{headwork_name}_河川水位"
                
                all_rate_columns.append(target_col)
                all_abs_columns.append(abs_col)

                if target_col not in df.columns:
                    self.logger.info(f"Computing target variables for horizon {horizon} minutes")
                    future_df = df[["datetime", water_level_col]].copy()
                    future_df["datetime"] -= pd.Timedelta(minutes=horizon)
                    future_df = future_df.rename(columns={water_level_col: "future_val"})

                    df = df.merge(future_df, on="datetime", how="left")
                    
                    df[target_col] = (df["future_val"] - df[water_level_col]) / df[water_level_col]
                    df[abs_col] = df["future_val"] - df[water_level_col]
                    df.drop(columns=["future_val"], inplace=True)

                if horizon in self.features_config[headwork_name]["lag_windows"]:
                    self.logger.info(f"Generating lag features for horizon {horizon} minutes")
                    for lag in self.features_config[headwork_name]["lag_windows"][horizon]:
                        target_lag_col = f"{target_col}_{lag}lag"
                        abs_lag_col = f"{abs_col}_{lag}lag"
                        
                        all_rate_columns.append(target_lag_col)
                        all_abs_columns.append(abs_lag_col)

                        lag_df = df[["datetime", target_col, abs_col]].copy()
                        lag_df["datetime"] += pd.Timedelta(minutes=lag)

                        df = df.merge(
                            lag_df.rename(columns={target_col: target_lag_col, abs_col: abs_lag_col}),
                            on="datetime", how="left"
                        )
        
        initial_rows = len(df)
        self.logger.info(f"Initial row count: {initial_rows}")
        
        original_datetimes = set(df['datetime'])
        
        self.logger.info("Filtering out NaN and Inf values in target columns")
        for col in all_rate_columns:
            if col in df.columns:
                df = df[~df[col].isna() & ~np.isinf(df[col])]
        
        self.logger.info("Filtering out unreasonable rate values (outside [-1, 1])")
        for col in all_rate_columns:
            if col in df.columns:
                df = df[(df[col] >= -1) & (df[col] <= 1)]
        
        removed_rows = initial_rows - len(df)
        self.logger.info(f"Removed {removed_rows} rows with NaN, Inf, or out-of-range values for {headwork_name}")
        
        remaining_datetimes = set(df['datetime'])
        removed_datetimes = original_datetimes - remaining_datetimes
        if removed_datetimes:
            self.logger.warning(f"Removed {len(removed_datetimes)} unique timestamps. First 5: {list(removed_datetimes)[:5]}")
        
        valid_features = self.get_features_for_headwork(headwork_name)
        valid_targets = [
            f"{horizon}分後変動率_{headwork_name}_河川水位"
            for horizon in self.features_config[headwork_name]["prediction_horizons"]
            if f"{horizon}分後変動率_{headwork_name}_河川水位" in df.columns
        ] + [
            f"{horizon}分後変動_{headwork_name}_河川水位"
            for horizon in self.features_config[headwork_name]["prediction_horizons"]
            if f"{horizon}分後変動_{headwork_name}_河川水位" in df.columns
        ]

        if water_level_col in df.columns:
            valid_targets.append(water_level_col)

        self.logger.info("Removing rows with NaN in features or targets")
        df.dropna(subset=valid_features + valid_targets)

        valid_features = [f for f in valid_features if f in df.columns]
        valid_targets = [t for t in valid_targets if t in df.columns]

        self.logger.info(f"Finding common non-NaN indices for {len(valid_features)} features and {len(valid_targets)} targets")
        common_index = df[valid_features].dropna().index.intersection(df[valid_targets].dropna().index)
        df = df.loc[common_index].reset_index(drop=True)
        self.logger.info(f"Final dataset size: {len(df)} rows")

        return df          



    def prepare_all_headworks(self, input_files=None):
        self.logger.info("Preparing data for all headworks")
        datasets = ["train", "val", "test"]

        if input_files is None:
            self.logger.info("Using default input files")
            input_files = {
                "train": self.ml_input_dir / "with_rainfall_train.csv",
                "val": self.ml_input_dir / "with_rainfall_val.csv",
                "test": self.ml_input_dir / "with_rainfall_test.csv"
            }

        os.makedirs(self.ml_featured_dir, exist_ok=True)
        self.logger.info(f"Output directory: {self.ml_featured_dir}")

        for headwork in self.features_config.keys():
            self.logger.info(f"Processing {headwork} data...")

            for dataset in datasets:
                if dataset in input_files and input_files[dataset].exists():
                    self.logger.info(f"- Preparing {dataset} dataset for {headwork}")
                    df = self.prepare_headwork_data(headwork, input_files[dataset])

                    features = self.get_features_for_headwork(headwork)

                    valid_features = [f for f in features if f in df.columns]
                    valid_targets = [
                        f"{horizon}分後変動率_{headwork}_河川水位"
                        for horizon in self.features_config[headwork]["prediction_horizons"]
                        if f"{horizon}分後変動率_{headwork}_河川水位" in df.columns
                    ] + [
                        f"{horizon}分後変動_{headwork}_河川水位"
                        for horizon in self.features_config[headwork]["prediction_horizons"]
                        if f"{horizon}分後変動_{headwork}_河川水位" in df.columns
                    ]

                    water_level_col = f"{headwork} 河川水位"
                    if water_level_col in df.columns:
                        valid_targets.append(water_level_col)

                    df_with_dt = df.copy()
                    
                    self.logger.info("Checking for NaN values in features and targets")
                    nan_features = {f: df[f].isna().sum() for f in valid_features if df[f].isna().sum() > 0}
                    nan_targets = {t: df[t].isna().sum() for t in valid_targets if df[t].isna().sum() > 0}
                    
                    if nan_features:
                        self.logger.warning(f"NaN values in features: {nan_features}")
                    if nan_targets:
                        self.logger.warning(f"NaN values in targets: {nan_targets}")
                    
                    self.logger.info("Splitting into X and y datasets")
                    df_x = df[valid_features + ["datetime"]]
                    df_y = df[valid_targets + ["datetime"]]
                    
                    self.logger.info("Verifying datetime alignment between X and y")
                    x_dates = set(df_x['datetime'])
                    y_dates = set(df_y['datetime'])
                    if x_dates != y_dates:
                        self.logger.warning(f"X and y have different timestamps!")
                        if len(x_dates - y_dates) > 0:
                            self.logger.warning(f"{len(x_dates - y_dates)} timestamps in X but not in y")
                        if len(y_dates - x_dates) > 0:
                            self.logger.warning(f"{len(y_dates - x_dates)} timestamps in y but not in X")
                    
                    df_x.reset_index(drop=True, inplace=True)
                    df_y.reset_index(drop=True, inplace=True)
                    
                    assert df_x.shape[0] == df_y.shape[0], "Mismatch between X and y after processing"
                    assert all(df_x['datetime'] == df_y['datetime']), "X and y datetime values don't match"

                    x_output_file = self.ml_featured_dir / f"{headwork}_{dataset}_X.csv"
                    y_output_file = self.ml_featured_dir / f"{headwork}_{dataset}_y.csv"
                    
                    df_x.to_csv(x_output_file, index=False, encoding='utf-8-sig')
                    df_y.to_csv(y_output_file, index=False, encoding='utf-8-sig')
                    self.logger.info(f"Saved {headwork}_{dataset} files: X={df_x.shape}, y={df_y.shape}")
                else:
                    self.logger.warning(f"Dataset file {dataset} not found for {headwork}")

        config_path = self.ml_featured_dir / "feature_config.json"
        self.logger.info(f"Saving feature configuration to {config_path}")

        if config_path.exists():
            self.logger.info("Updating existing configuration file")
            with open(config_path, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            existing_config.update(self.features_config)
        else:
            existing_config = self.features_config

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(existing_config, f, ensure_ascii=False, indent=2)

        self.logger.info("All headworks data prepared and stored in ml_featured!")





















        


