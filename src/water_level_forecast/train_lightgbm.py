import os
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import traceback
import logging
from datetime import datetime

class LightGBMTrainer:
    def __init__(self, processed_dir: str, target_headwork: str, with_rainfall: bool = True):
        self.processed_dir = Path(processed_dir)
        self.target_headwork = target_headwork
        self.with_rainfall = with_rainfall
        self.ml_featured_dir = self.processed_dir / "ml_featured"
        self.ml_output_dir = self.processed_dir / "ml_output"
        self.minmax_path = self.ml_output_dir / f"{self.target_headwork}_minmax.csv"
        self.forecast_horizons = [10, 20, 30, 40, 50, 60]
        os.makedirs(self.ml_output_dir, exist_ok=True)
        self.minmax = {}
        self.rate_models = {}
        self.abs_models = {}
        
        
        self.X_train_original = None
        self.X_val_original = None
        self.y_train_original = {}
        self.y_val_original = {}

        self.setup_logging()
        self.logger.info(f"LightGBMTrainer initialized for {self.target_headwork} with rainfall={with_rainfall}")

    def setup_logging(self):
        log_dir = self.processed_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{self.target_headwork}_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_train_params(self):
        params_dict = {
            "別所頭首工": {
                "metric": "l1",
                "num_leaves": 119,
                "max_depth": 9,
                "learning_rate": 0.02662681660026451,
                "n_estimators": 838,
                "subsample": 0.7646994957016992,
                "colsample_bytree": 0.6448917630775342,
                "min_child_samples": 41,
                "lambda_l1": 0.20163815067707308,
                "lambda_l2": 1.893778442228652e-05,
                "verbose": -1
            },
            "蒲生頭首工": {
                "metric": "l1",
                "num_leaves": 103,
                "max_depth": 6,
                "learning_rate": 0.039082987515792166,
                "n_estimators": 581,
                "subsample": 0.951746071412658,
                "colsample_bytree": 0.7055101281937456,
                "min_child_samples": 60,
                "lambda_l1": 0.7749706522381301,
                "lambda_l2": 8.94452604681556,
                "verbose": -1
            },
            "蓮花寺頭首工": {
                "metric": "l1",
                "num_leaves": 80,
                "max_depth": 10,
                "learning_rate": 0.01604524647433422,
                "n_estimators": 878,
                "subsample": 0.8534442854854237,
                "colsample_bytree": 0.5392368246127326,
                "min_child_samples": 57,
                "lambda_l1": 0.5659774303131794,
                "lambda_l2": 4.4234662429240676e-05,
                "verbose": -1
            },
            "名神日野川頭首工": { 
                "metric": "l1",
                "num_leaves": 43,
                "max_depth": 10,
                "learning_rate": 0.0300142962906802,
                "n_estimators": 779,
                "subsample": 0.9087645402576485,
                "colsample_bytree": 0.5517225948674603,
                "min_child_samples": 24,
                "lambda_l1": 0.007107059488707631,
                "lambda_l2": 5.4405198459602124e-05,
                "verbose": -1
            }
        }
        if self.target_headwork in params_dict:
            self.logger.info(f"Using optimized parameters for {self.target_headwork}")
            return params_dict[self.target_headwork]
        else:
            error_msg = f"{self.target_headwork} のトレーニングパラメータが見つかりません。頭首工の名前が正しいか確認してください。"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def normalize(self, series, column):
        """Normalize values using min-max scaling"""
        vmin = float(self.minmax[column]["min"])
        vmax = float(self.minmax[column]["max"])

        if isinstance(series, pd.Series):
            values = series.values
        else:
            values = series

        if vmax == vmin or np.isnan(vmax) or np.isnan(vmin):
            self.logger.warning(f"Adjusting normalization for '{column}': vmax ({vmax}) == vmin ({vmin}). Using small delta.")
            vmin -= 1e-6
  
        return (values - vmin) / (vmax - vmin)

    def denormalize(self, series, column):
        """Reverse normalization using min-max scaling"""
        vmin = float(self.minmax[column]["min"])
        vmax = float(self.minmax[column]["max"])

        if isinstance(series, pd.Series):
            values = series.values
        else:
            values = series
        
        if vmax == vmin or np.isnan(vmax) or np.isnan(vmin):
            self.logger.warning(f"Feature '{column}' has invalid vmax == vmin during denormalization. Adjusting.")
            vmin -= 1e-6 
            
        return (values * (vmax - vmin)) + vmin
    
    def back_check(self, x):
        """Categorize changes for data augmentation"""
        if abs(x) <= 0.005:
            return 0
        elif 0 < x <= 0.015:
            return 1
        elif -0.015 <= x < 0:
            return -1
        elif x > 0:
            return 2
        elif x < 0:
            return -2
    
    def prepare_data(self):
        """Load and prepare data for training"""
        self.logger.info("Preparing data...")
        X_train_path = self.ml_featured_dir / f"{self.target_headwork}_train_X.csv"
        y_train_path = self.ml_featured_dir / f"{self.target_headwork}_train_y.csv"
        X_val_path = self.ml_featured_dir / f"{self.target_headwork}_val_X.csv"
        y_val_path = self.ml_featured_dir / f"{self.target_headwork}_val_y.csv"

        if not X_train_path.exists() or not y_train_path.exists():
            error_msg = f"Training dataset files not found for {self.target_headwork}!"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.logger.info(f"Loading datasets from {self.ml_featured_dir}...")
        X_train = pd.read_csv(X_train_path, encoding='utf-8-sig', parse_dates=["datetime"])
        X_val = pd.read_csv(X_val_path, encoding='utf-8-sig', parse_dates=["datetime"])
        y_train_df = pd.read_csv(y_train_path, encoding='utf-8-sig', parse_dates=["datetime"])
        y_val_df = pd.read_csv(y_val_path, encoding='utf-8-sig', parse_dates=["datetime"])

        self.logger.info(f"Dataset sizes - X_train: {X_train.shape}, X_val: {X_val.shape}, y_train: {y_train_df.shape}, y_val: {y_val_df.shape}")

        for df in [X_train, X_val, y_train_df, y_val_df]:
            df.ffill(inplace=True)
            df.bfill(inplace=True)

   
        self.logger.info("Aligning timestamps between feature and target datasets...")
        common_train_time = pd.Index(X_train["datetime"]).intersection(pd.Index(y_train_df["datetime"]))
        common_val_time = pd.Index(X_val["datetime"]).intersection(pd.Index(y_val_df["datetime"]))

        self.logger.info(f"Common timestamps - Train: {len(common_train_time)}, Validation: {len(common_val_time)}")

        X_train = X_train[X_train["datetime"].isin(common_train_time)]
        y_train_df = y_train_df[y_train_df["datetime"].isin(common_train_time)]

        X_val = X_val[X_val["datetime"].isin(common_val_time)]
        y_val_df = y_val_df[y_val_df["datetime"].isin(common_val_time)]

        self.logger.info("Computing normalization parameters...")

        common_cols = set(X_train.columns) & set(y_train_df.columns)
        common_cols.discard("datetime")  # Keep datetime for merging
        if common_cols:
            self.logger.warning(f"Duplicate columns found: {common_cols}. Dropping them from y_train_df to avoid '_x' and '_y' suffixes.")
            y_train_df = y_train_df.drop(columns=common_cols)


        full_data = X_train.merge(y_train_df, on="datetime", how="inner")

        full_data = full_data.drop(columns=["datetime"])

        # Handle missing values
        full_data.ffill(inplace=True)
        full_data.bfill(inplace=True)

        full_data = full_data.apply(pd.to_numeric, errors='coerce').astype(float)

        if full_data.isna().sum().sum() > 0:
            self.logger.warning(f"Full dataset contains {full_data.isna().sum().sum()} NaN values after filling! Check preprocessing.")

        self.minmax = {}
        for col in full_data.columns:
            min_val = full_data[col].min()
            max_val = full_data[col].max()

            if np.isinf(max_val) or np.isinf(min_val):
                self.logger.warning(f"Found inf in minmax for {col}. Adjusting...")

            if np.isinf(max_val):
                max_val = abs(min_val)  
                self.logger.info(f"Adjusted max for {col}: {max_val}")


            if np.isnan(min_val) or np.isnan(max_val):
                self.logger.warning(f"Feature '{col}' has NaN min/max! Setting default range 0 ~ 1.")
                min_val, max_val = 0, 1

            # Prevent division by zero in normalization
            if min_val == max_val:
                self.logger.warning(f"Feature '{col}' has vmax ({max_val}) == vmin ({min_val}). Adjusting minmax.")
                max_val = min_val + 1e-6  

            self.minmax[col] = {"min": float(min_val), "max": float(max_val)}

        # Verify minmax dictionary
        # print("minmax keys:", self.minmax.keys())

        # Save minmax values
        minmax_df = pd.DataFrame.from_dict(self.minmax, orient="index").reset_index()
        minmax_df.columns = ["feature", "min", "max"]
        minmax_df.to_csv(self.minmax_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"Saved minmax normalization parameters to {self.minmax_path}")

        # Reload minmax file to verify correctness
        minmax_loaded = pd.read_csv(self.minmax_path, encoding="utf-8-sig")
        self.logger.info(f"Verified minmax file with {len(minmax_loaded)} features")



        X_train.sort_values(by="datetime", inplace=True)
        y_train_df.sort_values(by="datetime", inplace=True)
        X_val.sort_values(by="datetime", inplace=True)
        y_val_df.sort_values(by="datetime", inplace=True)

        self.logger.info("Data preparation completed successfully")
        return X_train, X_val, y_train_df, y_val_df

    def process_horizon(self, X_train, X_val, y_train_df, y_val_df, horizon):
        """Process a single forecast horizon"""
        self.logger.info(f"Processing horizon {horizon} minutes...")
        rate_col = f"{horizon}分後変動率_{self.target_headwork}_河川水位"
        abs_col = f"{horizon}分後変動_{self.target_headwork}_河川水位"

        if rate_col not in y_train_df.columns or abs_col not in y_train_df.columns:
            self.logger.warning(f"Skipping horizon {horizon} - target columns not found")
            return
        
        self.logger.info(f"Augmenting data for {horizon} minutes horizon...")

        y_train = y_train_df[["datetime", rate_col, abs_col]].rename(columns={
            rate_col: "rate", abs_col: "abs"
        })
        y_val = y_val_df[["datetime", rate_col, abs_col]].rename(columns={
            rate_col: "rate", abs_col: "abs"
        })

        train_df = X_train.merge(y_train, on="datetime", how="inner")
        val_df = X_val.merge(y_val, on="datetime", how="inner")

        train_df["check"] = train_df["abs"].apply(self.back_check)
        val_df["check"] = val_df["abs"].apply(self.back_check)

        self.logger.info(f"Applying data augmentation for horizon {horizon}...")
        train_df = self.augment_data(train_df)
        val_df = self.augment_data(val_df)
        self.logger.info(f"Data augmentation completed. Train shape: {train_df.shape}, Val shape: {val_df.shape}")


        minmax_mapping = {
        "rate": rate_col,  #
        "abs": abs_col     # 
        }

        self.logger.info(f"Normalizing features for horizon {horizon}...")
        feature_cols = [col for col in train_df.columns if col not in ["datetime", "check"]]
        for col in feature_cols:
            original_col = minmax_mapping[col] if col in minmax_mapping else col
            train_df[col] = self.normalize(train_df[col], original_col)
            val_df[col] = self.normalize(val_df[col], original_col)

 
        X_train = train_df.drop(columns=["rate", "abs", "datetime", "check"])
        y_train_rate = train_df["rate"]
        y_train_abs = train_df["abs"]

        X_val = val_df.drop(columns=["rate", "abs", "datetime", "check"])
        y_val_rate = val_df["rate"]
        y_val_abs = val_df["abs"]

        self.train_models(horizon, X_train, X_val, y_train_rate, y_val_rate, y_train_abs, y_val_abs)


        del train_df, val_df, X_train, X_val, y_train_rate, y_train_abs, y_val_rate, y_val_abs
        gc.collect()


    def augment_data(self, df):
        check1 = df[df["check"].isin([1, -1])].copy()
        check2 = df[df["check"].isin([2, -2])].copy()

        initial_size = len(df)

        for _ in range(3):
            df = pd.concat([df, check1], ignore_index=True)
        for _ in range(10):
            df = pd.concat([df, check2], ignore_index=True)

        df.reset_index(drop=True, inplace=True)

        self.logger.info(f"Data augmentation: {initial_size} rows → {len(df)} rows (x{len(df)/initial_size:.2f})")
        return df
    
    def train_models(self, horizon, X_train, X_val, y_train_rate, y_val_rate, y_train_abs, y_val_abs, max_rounds=1000):
        """Train and save models for a specific horizon"""
        self.logger.info(f"Training models for {horizon} minutes prediction with max_rounds={max_rounds}...")

        params = self.get_train_params()
        self.logger.info(f"Model parameters: {params}")

        try:
            self.logger.info(f"Training rate change model for {horizon} minutes...")
            rate_train = lgb.Dataset(X_train, label=y_train_rate)
            rate_val = lgb.Dataset(X_val, label=y_val_rate)

            rate_model = lgb.train(
                params,
                rate_train,
                valid_sets=[rate_val],
                num_boost_round=max_rounds,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True),
                           lgb.log_evaluation(50)]
            )

   
            model_path_rate = self.ml_output_dir / f"{self.target_headwork}_{horizon}min_rate.pkl"
            joblib.dump(rate_model, model_path_rate)
            self.logger.info(f"Rate model for {horizon} minutes saved to {model_path_rate}")

            del rate_train, rate_val
            gc.collect()

        except Exception as e:
            self.logger.error(f"Error training rate model for horizon {horizon}: {str(e)}")
            self.logger.error(traceback.format_exc())

        try:
            self.logger.info(f"Training absolute change model for {horizon} minutes...")
            abs_train = lgb.Dataset(X_train, label=y_train_abs)
            abs_val = lgb.Dataset(X_val, label=y_val_abs)

            abs_model = lgb.train(
                params,
                abs_train,
                valid_sets=[abs_val],
                num_boost_round=max_rounds,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True),
                           lgb.log_evaluation(50)]
            )


            model_path_abs = self.ml_output_dir / f"{self.target_headwork}_{horizon}min_abs.pkl"
            joblib.dump(abs_model, model_path_abs)
            self.logger.info(f"Absolute model for {horizon} minutes saved to {model_path_abs}")


            del abs_train, abs_val
            gc.collect()

        except Exception as e:
            self.logger.error(f"Error training absolute model for horizon {horizon}: {str(e)}")
            self.logger.error(traceback.format_exc())


    def run(self):
        """Run the complete training process"""
        try:
            self.logger.info(f"Starting training process for {self.target_headwork}...")

            X_train, X_val, y_train_df, y_val_df = self.prepare_data()

            for h in self.forecast_horizons:
                try:
                    self.process_horizon(X_train, X_val, y_train_df, y_val_df, h)
                    
                    gc.collect()

                except Exception as e:
                    error_msg = f"Error in processing horizon {h}: {str(e)}"
                    self.logger.error(error_msg)
                    self.logger.error(traceback.format_exc())

                    error_log_path = self.ml_output_dir / f"{self.target_headwork}_h{h}_error.log"
                    with open(error_log_path, 'w', encoding='utf-8') as f:
                        f.write(f"{error_msg}\n{traceback.format_exc()}")

            self.logger.info(f"Training completed successfully for {self.target_headwork}!")
            return True

        except Exception as e:
            error_msg = f"Error in overall processing: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())

            error_log_path = self.ml_output_dir / f"{self.target_headwork}_error.log"
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(f"{error_msg}\n{traceback.format_exc()}")

            return False
