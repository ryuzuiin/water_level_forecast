import os
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import gc
import logging
from datetime import datetime

class LightGBMPredictor:
    def __init__(self, processed_dir: str, target_headwork: str, with_rainfall: bool = True):
        self.processed_dir = Path(processed_dir)
        self.target_headwork = target_headwork
        self.with_rainfall = with_rainfall
        self.ml_featured_dir = self.processed_dir / "ml_featured"
        self.ml_output_dir = self.processed_dir / "ml_output"
        self.minmax_path = self.ml_output_dir / f"{self.target_headwork}_minmax.csv"
        self.forecast_horizons = [10, 20, 30, 40, 50, 60]
        os.makedirs(self.ml_output_dir, exist_ok=True)

        self.models = {}
        self.minmax = {}

        self.setup_logging()
        self.logger.info(f"LightGBMPredictor initialized for {self.target_headwork} with rainfall={with_rainfall}")


    def setup_logging(self):
        log_dir = self.processed_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"prediction_{self.target_headwork}_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    

    def load_minmax(self):
        self.logger.info(f"Loading MinMax normalization parameters...")
        if self.minmax_path.exists():
            path_to_use = self.minmax_path
        else:
            fallback_path = Path("models") / f"{self.target_headwork}_minmax.csv"
            if fallback_path.exists():
                self.logger.warning(f"MinMax file not found in default location. Using fallback: {fallback_path}")
                path_to_use = fallback_path
            else:
                error_msg = f"Min-Max file not found at either:\n  - {self.minmax_path}\n  - {fallback_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        minmax_df = pd.read_csv(path_to_use, encoding="utf-8-sig")
        self.minmax = minmax_df.set_index("feature").to_dict(orient="index")
        self.logger.info(f"MinMax parameters loaded successfully with {len(minmax_df)} features")


    def load_models(self):
        self.logger.info(f"Loading prediction models for {len(self.forecast_horizons)} horizons...")
        for horizon in self.forecast_horizons:
            
            default_model_path = self.ml_output_dir / f"{self.target_headwork}_{horizon}min_rate.pkl"
            

            if default_model_path.exists():
                model_path = default_model_path
            else:
                fallback_path = Path("models") / f"{self.target_headwork}_{horizon}min_rate.pkl"
                if fallback_path.exists():
                    self.logger.warning(f"Default model not found for {horizon}min. Using fallback model from: {fallback_path}")
                    model_path = fallback_path
                else:
                    error_msg = f"Model file not found at either:\n  - {default_model_path}\n  - {fallback_path}"
                    self.logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
            
            self.models[horizon] = joblib.load(model_path)
            self.logger.info(f"Model for {horizon}min horizon loaded from {model_path}")


    def load_test_data(self):
        self.logger.info("Loading test data...")
        X_test_path = self.ml_featured_dir / f"{self.target_headwork}_test_X.csv"
        y_test_path = self.ml_featured_dir / f"{self.target_headwork}_test_y.csv"

        fallback_path = Path("data") / "test"
        fallback_X = fallback_path / f"{self.target_headwork}_test_X.csv"
        fallback_y = fallback_path / f"{self.target_headwork}_test_y.csv"

        if X_test_path.exists() and y_test_path.exists():
            path_X, path_y = X_test_path, y_test_path

        elif fallback_X.exists() and fallback_y.exists():
            self.logger.warning(f"Default test data not found. Using fallback from: {fallback_path}")
            path_X, path_y = fallback_X, fallback_y
        else:
            error_msg = f"Test dataset not found at either:\n  - {X_test_path}\n  - {fallback_X}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.logger.info(f"Loading test data from:\n  - X: {path_X}\n  - y: {path_y}")
        

        X_test = pd.read_csv(path_X, encoding='utf-8-sig', parse_dates=["datetime"])
        y_test = pd.read_csv(path_y, encoding='utf-8-sig', parse_dates=["datetime"])
        self.logger.info(f"Test data loaded - X shape: {X_test.shape}, y shape: {y_test.shape}")

        self.logger.info("Aligning timestamps between X and y test datasets...")

        common_time = pd.Index(X_test["datetime"]).intersection(pd.Index(y_test["datetime"]))
        X_test = X_test[X_test["datetime"].isin(common_time)].copy()
        y_test = y_test[y_test["datetime"].isin(common_time)].copy()
        self.logger.info(f"Common timestamps: {len(common_time)}")

   
        self.datetime_col = X_test[["datetime"]].copy()
        self.water_level_col = f"{self.target_headwork} 河川水位"

        if self.water_level_col in y_test.columns:
            self.water_level_values = y_test[self.water_level_col].copy()
            self.logger.info(f"Found water level column: {self.water_level_col}")
        else:
            error_msg = f"Cannot find water level column '{self.water_level_col}' in y_test!"
            self.logger.error(error_msg)
            raise KeyError(error_msg)


        self.X_test = X_test.drop(columns=["datetime"], errors='ignore')
        self.y_test = y_test.drop(columns=["datetime"], errors='ignore')

        self.logger.info("Checking X_test for NaN before normalization...")
        nan_count = self.X_test.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"X_test contains {nan_count} NaN values! Filling with zeros.")
            self.X_test.fillna(0, inplace=True) 


    def normalize(self, series, column):
     
        vmin = self.minmax[column]["min"]
        vmax = self.minmax[column]["max"]

        if np.isnan(vmin) or np.isnan(vmax):
            self.logger.warning(f"vmin or vmax is NaN for {column}! Setting default range (0,1).")
            vmin, vmax = 0, 1

        if vmax == vmin:
            self.logger.warning(f"vmax == vmin ({vmax}) for {column}. Adjusting to avoid division by zero.")
            vmax = vmin + 1e-6  

        normalized_values = (series - vmin) / (vmax - vmin)

        if np.isnan(normalized_values).sum() > 0:
            self.logger.warning(f"NaN found in normalize() for {column}! vmin: {vmin}, vmax: {vmax}")
    
        return normalized_values
        


    def denormalize(self, series, column):
        vmin = self.minmax[column]["min"]
        vmax = self.minmax[column]["max"]

        if np.isnan(vmin) or np.isnan(vmax):
            self.logger.warning(f"vmin or vmax is NaN for {column}!")
            return np.zeros_like(series)

        if vmax == vmin:
            self.logger.warning(f"vmax == vmin ({vmax}) for {column}. Adjusting to avoid division by zero.")
            vmax = vmin + 1e-6

        denormalized_values = (series * (vmax - vmin)) + vmin

        if np.isnan(denormalized_values).sum() > 0:
            self.logger.warning(f"NaN found in denormalized {column}!")
    
        return denormalized_values

    def back_check(self, x):
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
        

    def predict(self):
        self.logger.info(f"Starting predictions for {len(self.forecast_horizons)} horizons...")
        y_pred = np.zeros((self.X_test.shape[0], len(self.forecast_horizons)))

        if "datetime" in self.X_test.columns:
            self.X_test = self.X_test.drop(columns=["datetime"])

        self.logger.info("Normalizing test features...")
        X_test_normalized = self.X_test.copy()
        feature_cols = list(X_test_normalized.columns)
        for col in feature_cols:
            X_test_normalized[col] = self.normalize(X_test_normalized[col], col)

        if np.isnan(X_test_normalized.to_numpy()).sum() > 0:
            self.logger.warning(f"X_test_normalized contains NaN after normalization!")

        for i, horizon in enumerate(self.forecast_horizons):
            self.logger.info(f"Predicting for horizon {horizon} minutes...")
            y_pred[:, i] = self.models[horizon].predict(self.X_test)

        if np.isnan(y_pred).sum() > 0:
            self.logger.warning(f"LightGBM predicted {np.isnan(y_pred).sum()} NaN values!")

        self.logger.info("Creating results dataframe...")
        results = pd.DataFrame(self.datetime_col)
        results[self.water_level_col] = self.water_level_values

        for i, horizon in enumerate(self.forecast_horizons):
            target_col = f"{horizon}分後変動率_{self.target_headwork}_河川水位"
            abs_col = f"{horizon}分後変動_{self.target_headwork}_河川水位"

            self.logger.info(f"Processing results for horizon {horizon} minutes...")
            rate_pred = self.denormalize(y_pred[:, i], target_col)

            actual_rate = self.y_test[target_col] if target_col in self.y_test.columns else None

            pred_absolute_change = self.water_level_values * rate_pred
            actual_absolute_change = self.y_test[abs_col] if abs_col in self.y_test.columns else None

            results[f"{horizon}分後変動率_pred"] = rate_pred
            if actual_rate is not None:
                results[f"{horizon}分後変動率_actual"] = actual_rate
            results[f"{self.target_headwork}_{horizon}分後水位"] = self.water_level_values + pred_absolute_change

            if actual_absolute_change is not None:
                results[f"check_actual_{horizon}"] = actual_absolute_change.apply(self.back_check)

        self.logger.info("Calculating MAE by category...")
        category_mae = {h: {} for h in self.forecast_horizons}
        categories = [-2, -1, 0, 1, 2]

        for h in self.forecast_horizons:
            target_col = f"{h}分後変動率_{self.target_headwork}_河川水位"
            pred_col = f"{h}分後変動率_pred"
            actual_col = f"{h}分後変動率_actual"
            check_actual_col = f"check_actual_{h}"

            if pred_col not in results.columns:
                self.logger.warning(f"Skipping {h} minutes: Missing target or prediction columns.")
                continue

            sf = 259 * (float(self.minmax[target_col]["max"]) - float(self.minmax[target_col]["min"]))
            self.logger.info(f"Processing horizon {h} minutes: Scale Factor = {sf}")

            if np.isinf(sf):
                self.logger.warning(f"Scale Factor for {h} minutes is Inf! Check minmax values.")

            for cat in categories:
                mask = results[check_actual_col] == cat
                if mask.sum() > 0:
                    category_mae[h][cat] = sf * mean_absolute_error(
                        results.loc[mask, actual_col],
                        results.loc[mask, pred_col]
                    )
                else:
                    category_mae[h][cat] = None
                self.logger.info(f"Category {cat}: {mask.sum()} samples")

        self.logger.info("Saving results...")
        mae_df = pd.DataFrame(category_mae).T
        mae_output_file = self.ml_output_dir / f"{self.target_headwork}_category_mae_cm.csv"
        mae_df.to_csv(mae_output_file, encoding="utf-8-sig")
        self.logger.info(f"MAE by category saved to {mae_output_file}")

        output_file = self.ml_output_dir / f"{self.target_headwork}_predictions.csv"
        results.to_csv(output_file, encoding="utf-8-sig", index=False)
        self.logger.info(f"Predictions saved to {output_file}")

    def run(self):
        self.logger.info(f"Starting prediction process for {self.target_headwork}...")
        self.load_minmax()
        self.load_models()
        self.load_test_data()
        self.predict()
        self.logger.info(f"Prediction completed successfully for {self.target_headwork}!")
        gc.collect()