import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

class MLInputPreparation:
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.cleaned_dir = self.processed_dir / "cleaned"
        self.anomaly_dir = self.processed_dir / "anomaly_cleaned"
        self.ml_input_dir = self.processed_dir / "ml_input"
       
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        log_dir = self.processed_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ml_input_preparation_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file,encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        self.ml_input_dir.mkdir(parents=True, exist_ok=True)

    def fill_missing_values(self, df: pd.DataFrame, file_name: str):
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            self.logger.warning(f"Detected {missing_count} missing values in {file_name}. Filling missing values...")
            df.fillna(method="ffill", inplace=True)
            df.fillna(method="bfill", inplace=True)
            missing_after = df.isna().sum().sum()
            self.logger.info(f"Missing values in {file_name} filled successfully. Remaing missing: {missing_after}")
        else:
            self.logger.info(f"No missing values in {file_name}.")

        return df
    

    def merge_rainfall_data(self):
        headworks_file = self.anomaly_dir / "headworks_cleaned.csv"
        if not headworks_file.exists():
            self.logger.error(f"Missing file: {headworks_file}")
            raise FileNotFoundError(f"Missing file: {headworks_file}")
        
        self.logger.info(f"Reading file: {headworks_file}")
        df_headworks = pd.read_csv(headworks_file, encoding='utf-8-sig', parse_dates=['datetime'])
        df_headworks = self.fill_missing_values(df_headworks, "headworks_cleaned.csv")

        rainfall_files = list(self.cleaned_dir.glob("*雨量*.csv"))
        if not rainfall_files:
            self.logger.error("No rainfall data files found in the cleaned directory!")
            raise FileNotFoundError("No rainfall data files found in the cleaned directory!")

        self.logger.info(f"Found {len(rainfall_files)} rainfall data files. Merging...")
        
        
        all_rainfall_data = []
        for file in rainfall_files:
            self.logger.info(f"Reading rainfall data: {file}")
            df_rainfall = pd.read_csv(file, encoding='utf-8-sig', parse_dates=["datetime"])
            df_rainfall = self.fill_missing_values(df_rainfall, file.name)
            all_rainfall_data.append(df_rainfall)
        
        df_rainfall_merged = pd.concat(all_rainfall_data, ignore_index=True).drop_duplicates()
        df_merged = pd.merge(df_headworks, df_rainfall_merged, on="datetime", how="left")

        df_merged.drop(columns=[col for col in df_merged.columns if "soure_file" in col], inplace=True)

        df_merged = self.fill_missing_values(df_merged, "with_rainfall.csv")

        with_rainfall_file = self.ml_input_dir / "with_rainfall.csv"
        df_merged.to_csv(with_rainfall_file, encoding="utf-8-sig", index=False)
        self.logger.info(f"Saved file: {with_rainfall_file}")

        
        without_rainfall_file = self.ml_input_dir / "without_rainfall.csv"
        df_headworks.to_csv(without_rainfall_file, encoding='utf-8-sig', index=False)
        self.logger.info(f"Saved file: {without_rainfall_file}")
        
    def split_data(self, input_file: Path, output_prefix: str):
        if not input_file.exists():
            self.logger.error(f"Missing file: {input_file}")
            raise FileNotFoundError(f"Missing file: {input_file}")
        
        self.logger.info(f"Reading file: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8-sig', parse_dates=["datetime"])

        df.drop(columns=[col for col in df.columns if "source_file" in col], inplace=True)

        df = self.fill_missing_values(df, input_file.name)
        
        train_size = 0.75
        val_size = 0.15
        test_size = 0.15
        
        train, temp = train_test_split(df, test_size=(1 - train_size), shuffle=False)
        val, test = train_test_split(temp, test_size=(test_size / (val_size + test_size)), shuffle=False)

        train_file = self.ml_input_dir / f"{output_prefix}_train.csv"
        val_file = self.ml_input_dir / f"{output_prefix}_val.csv"
        test_file = self.ml_input_dir / f"{output_prefix}_test.csv"
        
        train.to_csv(train_file, encoding='utf-8-sig', index=False)
        val.to_csv(val_file, encoding='utf-8-sig', index=False)
        test.to_csv(test_file, encoding='utf-8-sig', index=False)

        self.logger.info(f"Train dataset saved: {train_file}")
        self.logger.info(f"Validation dataset saved: {val_file}")
        self.logger.info(f"Test dataset saved: {test_file}")
        
    def process_all(self):

        self.logger.info("Starting ML input data preparation...")
        self.merge_rainfall_data()
        self.split_data(self.ml_input_dir / "with_rainfall.csv", "with_rainfall")
        self.split_data(self.ml_input_dir / "without_rainfall.csv", "without_rainfall")
        self.logger.info("ML input data preparation completed!")

    
    def close_logger(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.shutdown()
