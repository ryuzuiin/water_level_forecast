import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path
from datetime import datetime
import re

class DataCleaner:
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.filtered_dir = self.processed_dir / "filtered"  
        self.cleaned_dir = self.processed_dir / "cleaned"
        self.stats_dir = self.processed_dir / "stats"
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        """Set up logging"""
        log_dir = self.processed_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'cleaning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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
        """Create necessary directories"""
        for directory in [self.cleaned_dir, self.stats_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def detect_and_fix_anomalies(self, df):
        anomalies = {}

        for col in df.columns:
            if col in ['datetime', 'Year', 'Month', 'Day', 'Hour', 'Minute']:
                continue

            anomalies[col] = {
                'total': len(df[col]),
                'null': df[col].isnull().sum(),
                'zero': (df[col] == 0).sum(),
                'asterisk': df[col].astype(str).str.strip().isin(['*', '＊']).sum(),
                'garbled': df[col].astype(str).apply(lambda x: len(re.findall(r'[^\x00-\x7F]+', str(x))) if pd.notnull(x) else 0).sum(),
                'unique': df[col].nunique()
            }

            df[col] = pd.to_numeric(df[col], errors='coerce')

        anomalies_df = pd.DataFrame(anomalies).T

        stats_dir = self.stats_dir
        stats_dir.mkdir(parents=True, exist_ok=True)
        anomalies_df.to_csv(stats_dir / "anomalies_statistics.csv", encoding="utf-8-sig")

        return df  # Only return the cleaned df

    def generate_yearly_statistics(self, df, time_col, value_cols):
        stats_dir = self.stats_dir
        stats_dir.mkdir(parents=True, exist_ok=True)

        for col in value_cols:
            for year, group in df.groupby(df[time_col].dt.year):
                plt.figure(figsize=(10, 5))
                plt.plot(group[time_col], group[col], label=f"{col} - {year}")
                plt.xlabel("時間")
                plt.ylabel("値")
                plt.title(f"{col} の年間推移 ({year})")
                plt.legend()
                plt.xticks(rotation=45)
                plt.savefig(stats_dir / f"{col}_{year}_trend.png", bbox_inches="tight")
                plt.close()

                # Create yearly distribution chart
                plt.figure(figsize=(10, 5))
                plt.hist(group[col].dropna(), bins=30, edgecolor='black')
                plt.xlabel("値")
                plt.ylabel("頻度")
                plt.title(f"{col} 的年間分布 ({year})")
                plt.savefig(stats_dir / f"{col}_{year}_distribution.png", bbox_inches="tight")
                plt.close()

    def clean_file(self, file_path: Path):
        """Clean a single CSV file"""
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)

            df = self.detect_and_fix_anomalies(df)

            if '雨量' in file_path.name:  
                df = self.process_rainfall_data(df, file_path.name)

            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

        
            df = df[df["datetime"].dt.minute.isin([0, 10, 20, 30, 40, 50])]

            df = df.sort_values(by="datetime")

            output_path = self.cleaned_dir / file_path.name
            df.to_csv(output_path, encoding="utf-8-sig", index=False)
            self.logger.info(f"Cleaned and saved: {output_path}")

        
            self.generate_yearly_statistics(df, "datetime", df.columns)

            return True

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return False

    def clean_all_files(self):
        """Clean all CSV files in the filtered directory."""
        files = list(self.filtered_dir.glob("*.csv")) 

        if not files:
            self.logger.warning("No files found in filtered directory. Please check the previous processing steps.")
            return

        self.logger.info(f"Found {len(files)} CSV files to clean.")
        self.logger.info(f"Cleaning the following files: {[f.name for f in files]}")

        for file in files:
            result = self.clean_file(file)
            if result:
                self.logger.info(f"Successfully cleaned: {file.name}")
            else:
                self.logger.warning(f"Failed to clean: {file.name}")

        self.logger.info(f"Cleaning completed. Total cleaned files: {len(files)}.")

    def process_rainfall_data(self, df, file_name):
        """Process rainfall data by upsampling to 10-minute intervals."""
        if '雨量' in file_name:  
            df['datetime'] = pd.to_datetime(df['datetime']) 
            df.set_index('datetime', inplace=True) 

            #Notice here! make sure that rainfall data is within April and September again ! 
            df_resampled = df.resample('10T').asfreq()
            df_resampled = df_resampled[(df_resampled.index.month >= 4) &(df_resampled.index.month <= 9)]
            df_resampled.fillna(0, inplace=True)

            for col in df.columns:
                if col != 'datetime':  
                    original_hourly_data = df[col]

                    for i in range(len(original_hourly_data)):
                        if original_hourly_data[i] == 0:
                            df_resampled.iloc[i*6:(i+1)*6, df_resampled.columns.get_loc(col)] = 0
                        else:
                            hour_rainfall = original_hourly_data[i]
                            df_resampled.iloc[i*6:(i+1)*6, df_resampled.columns.get_loc(col)] = hour_rainfall / 6

            df_resampled.reset_index(inplace=True)

            return df_resampled
        return df
