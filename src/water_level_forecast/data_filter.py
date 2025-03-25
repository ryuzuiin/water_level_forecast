import os
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

class DataFilter:
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.filtered_dir = self.processed_dir / "filtered"  
        self.merged_dir = self.processed_dir / "merged"
        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        """Set up logging"""
        log_dir = self.processed_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'filtering_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.log'

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
        self.filtered_dir.mkdir(parents=True, exist_ok=True)
        # Ensure the merged directory exists
        if not self.merged_dir.exists():
            self.logger.warning(f"Warning: The directory {self.merged_dir} does not exist!")

    def combine_date_time(self, df: pd.DataFrame):
        """Combine date and time columns into one datetime column and convert it"""
        try:
            df["日付"] = df["日付"].fillna("")
            df["時刻"] = df["時刻"].fillna("")
            # Ensure that both date and time columns are valid
            df["datetime"] = df.apply(lambda row: f"{row['日付']} {row['時刻']}" if row["日付"] and row["時刻"] else None, axis=1)
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            return df.drop(columns=["日付", "時刻"])
        except Exception as e:
            self.logger.error(f"Error combining date and time: {e}")
            return df

    def filter_data(self, df: pd.DataFrame):
        """Filter data for the following criteria:
        1. Only data from 2018 onwards
        2. Data from April to September each year
        """
        # Filter for dates after 2018
        df = df[df["datetime"].dt.year >= 2018]

        # Filter for data from April to September
        df = df[(df["datetime"].dt.month >= 4) & (df["datetime"].dt.month <= 9)]

        return df

    def process_file(self, file_path: Path):
        """Process a single file and save the filtered data"""
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", dtype=str)

            # Combine date and time
            df = self.combine_date_time(df)

            # If the datetime column was created successfully, proceed
            if "datetime" not in df.columns:
                self.logger.warning(f"No valid datetime column found in {file_path}")
                return False

            # Sort by datetime
            df = df.sort_values(by="datetime")

            # Filter the data (2018 onwards, April to September)
            df = self.filter_data(df)

            # Save the filtered data to the filtered folder
            output_path = self.filtered_dir / file_path.name
            df.to_csv(output_path, encoding="utf-8-sig", index=False)
            self.logger.info(f"Processed and saved: {output_path}")

            return True
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return False

    def process_all_files(self):
        """Process all CSV files in the merged directory"""
        files = list(self.merged_dir.glob("*.csv"))

        if not files:
            self.logger.warning("No files found in the merged directory.")
            return

        self.logger.info(f"Found {len(files)} CSV files to process.")
        for file in files:
            self.process_file(file)

