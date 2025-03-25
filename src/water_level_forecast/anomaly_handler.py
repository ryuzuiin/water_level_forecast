import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.ensemble import IsolationForest
from datetime import datetime

class AnomalyHandler:
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.cleaned_dir = self.processed_dir / "cleaned"
        self.anomaly_dir = self.processed_dir / "anomaly_cleaned"
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        log_dir = self.processed_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"anomaly_cleaning_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        for directory in [self.cleaned_dir, self.anomaly_dir]:
            directory.mkdir(parents=True, exist_ok=True)


    def process_file(self, file_path):
        try:
            self.logger.info(f"Processing file: {file_path}")
            df = pd.read_csv(file_path, encoding = 'utf-8-sig')

            if "datetime" not in df.columns:
                self.logger.warning(f"File {file_path} is missing datetime column")
                return None
            
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

            if df["datetime"].isna().any():
                self.logger.warning(f"Some datetime values in file {file_path} are invalid")
                df = df.dropna(subset=["datetime"])

            df.drop(columns=[col for col in df.columns if "source_file" in col], inplace=True)

            self.logger.info(f"Successfully processed file {file_path}, containing {len(df)} rows")
            return df
        
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

   

    def detect_anomalies(self, df, column, contamination=0.05):
        self.logger.info(f"Detecting anomalies in column {column} with contamination={contamination}")
        df_copy = df.copy()

        if df_copy[column].isna().sum() > 0:
            self.logger.info(f"Filling NaN values in column {column} before anomaly detection")
            df_copy[column] = df_copy[column].interpolate(method="linear").ffill().bfill()
        model = IsolationForest(contamination=contamination, random_state=42)
        df_copy["anomaly_score"] = model.fit_predict(df_copy[[column]])

        
        anomalies = df_copy[df_copy["anomaly_score"] == -1].copy()
        self.logger.info(f"Found {len(anomalies)} anomalies in column {column}")

        return anomalies

    def remove_and_fill_anomalies(self, df, column, contamination=0.03):
        if column == "datetime":
            self.logger.info("Skipping datetime column for anomaly detection")
            return df
        
        self.logger.info(f"Removing and filling anomalies in column {column}")
        df_copy = df.copy()

        if "datetime" in df_copy.columns:
            df_copy["datetime"] = pd.to_datetime(df_copy["datetime"], errors="coerce")
            df_copy = df_copy.set_index("datetime")
        else:
            self.logger.warning(f"No datetime column found for time-based interpolation in {column}")

        anomalies = self.detect_anomalies(df_copy, column, contamination)
        self.logger.info(f"Detected {len(anomalies)} anomalies in column {column} (out of {len(df_copy)} rows)")

        df_copy.loc[anomalies.index, column] = np.nan
        self.logger.info(f"Filling {len(anomalies)} anomalies in column {column} with interpolation")

        try:
            df_copy[column] = df_copy[column].interpolate(method="time").ffill().bfill()
        except Exception as e:
            self.logger.warning(f"Time interpolation failed for {column}, using linear interpolation instead: {e}")
            df_copy[column] = df_copy[column].interpolate(method="linear").ffill().bfill()

        df_copy = df_copy.reset_index()
        self.logger.info(f"Anomaly handling completed for column {column}")

        return df_copy

    def process_all_files(self):
        self.logger.info("Starting water level data processing...")

        merged_df = self._collect_and_merge_data()
        if merged_df is None:
            self.logger.error("Failed to collect and merge data")
            return False
        
        headworks_file_path = self.anomaly_dir / "headwork.csv"
        try:
            merged_df.to_csv(headworks_file_path, encoding='utf-8-sig', index=False)
            self.logger.info(f"Saved merged raw file: {headworks_file_path}")
        except Exception as e:
            self.logger.error(f"Error saving file: {str(e)}")
            return False
        
        water_level_columns = [col for col in merged_df.columns if '水位' in col]
        self.logger.info(f"Processing {len(water_level_columns)} water level columns: {water_level_columns}")

        for column in water_level_columns:
            try:
                self.logger.info(f"Cleaning anomalies in column: {column}")
                merged_df[column] = self.remove_and_fill_anomalies(merged_df, column)[column]
            except Exception as e:
                self.logger.error(f"Error processing column {column}:{str(e)}")

        
        # step4: Save Cleaned data
        cleaned_file_path = self.anomaly_dir / "headworks_cleaned.csv"
        try:
            merged_df.to_csv(cleaned_file_path, encoding='utf-8-sig', index=False)
            self.logger.info(f"Saved cleaned file: {cleaned_file_path}")
        except Exception as e:
            self.logger.error(f"Error saving cleaned file: {str(e)}")
            return False
        
        self.logger.info("Anomaly processing completed")
        return True
    

    def _collect_and_merge_data(self):
        self.logger.info("Collecting and merging data files...")
        # all_data = []

        matching_files = list(self.cleaned_dir.glob("*頭首工*.csv"))
        self.logger.info(f"Found {len(matching_files)} matching files")

        if not matching_files:
            self.logger.warning("No matching files found. Check directory and file name pattern.")
            return None
        
        merged_df = None
        
        for file_path in matching_files:
            df = self.process_file(file_path)
            if df is None:
                continue

            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on="datetime", how="outer")

        if merged_df is None or "datetime" not in merged_df.columns:
            self.logger.error("ERROR: 'datetime column is missing after merging! check input files.")
            return None


        merged_df = merged_df.sort_values("datetime").reset_index(drop=True)

        self.logger.info(f"Sucessfully merged data, {len(merged_df)} rowa and {len(merged_df.columns)} columns")

        return merged_df
    

    def close_logger(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.shutdown()




