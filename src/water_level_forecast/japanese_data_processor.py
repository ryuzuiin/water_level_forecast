import os
import shutil
import zipfile
import logging
import chardet
import pandas as pd
import numpy as np
import io
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple


class JapaneseDataProcessor:
    def __init__(self, input_dir: str, output_dir: str, max_workers: int = 4):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.unzip_dir = self.output_dir / 'unzip'
        self.merged_dir = self.output_dir / 'merged'
        self.filtered_dir = self.output_dir / 'filtered'
        self.encoding_report_dir = self.output_dir / 'encoding_reports'

        self.max_workers = max_workers
        self.chunk_size = 50000  # Set larger chunk size for better performance

        self.setup_logging()
        self.setup_directories()

    def setup_logging(self):
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        for directory in [self.unzip_dir, self.merged_dir, self.filtered_dir, self.encoding_report_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_processed_files(self) -> List[str]:
        return [f.name for f in self.unzip_dir.iterdir() if f.is_dir()]

    def process_single_zip(self, zip_path: Path) -> List[Dict]:
        encoding_info = []
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith('.csv'):
                        with zip_ref.open(file_info.filename) as csv_file:
                            content = csv_file.read(1024)
                            result = chardet.detect(content)
                            encoding_info.append({
                                'zip_file': zip_path.name,
                                'csv_file': file_info.filename,
                                'detected_encoding': result['encoding'],
                                'confidence': result['confidence']
                            })
        except Exception as e:
            self.logger.error(f"Error analyzing {zip_path}: {e}")
        return encoding_info

    def analyze_zip_encodings(self):
        zip_files = list(self.input_dir.glob("*.zip"))
        self.logger.info(f"Found {len(zip_files)} ZIP files")

        encoding_report = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_zip = {executor.submit(self.process_single_zip, zip_path): zip_path for zip_path in zip_files}
            for future in as_completed(future_to_zip):
                encoding_report.extend(future.result())

        report_df = pd.DataFrame(encoding_report)
        report_path = self.encoding_report_dir / 'encoding_analysis.csv'
        report_df.to_csv(report_path, encoding='utf-8-sig', index=False)
        self.logger.info(f"Saved encoding report to {report_path}")

    def unzip_and_convert(self):
        zip_files = list(self.input_dir.glob("*.zip"))

        for zip_path in zip_files:
            # output_folder = self.unzip_dir / zip_path.stem
            # output_folder.mkdir(parents=True, exist_ok=True)

            # self.logger.info(f"Extracting {zip_path} to {output_folder}")

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for file_info in zip_ref.infolist():
                        try:
                            filename = file_info.filename.encode('cp437').decode('cp932')
                        except UnicodeDecodeError:
                            filename = file_info.filename

                        
                        if not  filename.lower().endswith('.csv'):
                            continue

                        filename = Path(filename).name

                        extracted_file_path = self.unzip_dir / filename

                        with zip_ref.open(file_info.filename) as src, open(extracted_file_path, 'wb') as dest:
                            shutil.copyfileobj(src,dest)

                        self.logger.info(f"✅ 解凍成功: {extracted_file_path}")

            except Exception as e:
                self.logger.error(f"Error processing {zip_path}: {e}")

    def process_single_csv(self, zip_path: Path, file_info: zipfile.ZipInfo) -> Optional[Tuple[str, pd.DataFrame]]:
        try:
            original_filename = file_info.filename.encode('cp437').decode('cp932')
            if not original_filename.endswith('.csv'):
                return None

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                with zip_ref.open(file_info) as csv_file:
                    content = csv_file.read()
                    encodings = ['cp932', 'shift_jis', 'utf-8', 'euc_jp']

                    for encoding in encodings:
                        try:
                            text_content = content.decode(encoding)
                            df = pd.read_csv(io.StringIO(text_content), dtype=str, on_bad_lines='skip', quoting=3, engine='python')
                            return original_filename, df
                        except Exception:
                            continue
            return None
        except Exception as e:
            self.logger.error(f"Error processing {file_info.filename} in {zip_path}: {e}")
            return None


    def merge_files(self):
        csv_files = list(self.unzip_dir.glob("*.csv"))
        file_groups = {}

        for file_path in csv_files:
            base_name = file_path.stem[:-4] if file_path.stem[-4:].isdigit() else file_path.stem
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_path)

        for base_name, files in file_groups.items():
            try:
                files.sort(key=lambda x: x.stem[-4:])
                
                dfs = []
                all_columns = set()
                
                for file in files:
                    encodings = ['utf-8', 'shift_jis', 'cp932', 'euc_jp']
                    for encoding in encodings:
                        try:
                            df = pd.read_csv(
                                file, 
                                encoding=encoding, 
                                dtype=str,
                                on_bad_lines='skip', 
                                quoting=3, 
                                engine='python',
                                # error_bad_lines=False
                            )
                            
                            df.columns = [str(col).strip() for col in df.columns]
                            all_columns.update(df.columns)
                            df['source_file'] = file.name

                            self.logger.info(f"File {file.name} has {len(df.columns)} columns")
                            
                            dfs.append(df)
                            break  
                        except Exception as e:
                            self.logger.warning(f"Failed to read {file} with {encoding} encoding: {e}")
                            continue  
                if not dfs:
                    self.logger.error(f"No files for {base_name} were successfully read")
                    continue
                    
                try:
                    # Try direct merge
                    merged_df = pd.concat(dfs, ignore_index=True)
                    output_path = self.merged_dir / f"{base_name}_merged.csv"
                    merged_df.to_csv(output_path, encoding='utf-8-sig', index=False)
                    self.logger.info(f"Successfully merged {len(files)} files into {output_path}")
                except Exception as e:
                    self.logger.warning(f"Standard merge method failed for {base_name}: {e}")
                    
                    # Try alternative method: merge using common columns
                    self.logger.info("Attempting to merge using common columns...")
                    try:
                        # Find columns common to all dataframes
                        common_columns = set(dfs[0].columns)
                        for df in dfs[1:]:
                            common_columns = common_columns.intersection(set(df.columns))
                        
                        # Add source file column
                        common_columns.add('source_file')
                        
                        # Keep only common columns
                        for i in range(len(dfs)):
                            dfs[i] = dfs[i][list(common_columns)]
                        
                        # Merge again
                        merged_df = pd.concat(dfs, ignore_index=True)
                        output_path = self.merged_dir / f"{base_name}_merged_common_cols.csv"
                        merged_df.to_csv(output_path, encoding='utf-8-sig', index=False)
                        self.logger.info(f"Successfully merged {len(files)} files into {output_path} using common columns (kept {len(common_columns)} columns)")
                    except Exception as e2:
                        self.logger.error(f"All merge methods failed for {base_name}: {e2}")
                    
            except Exception as e:
                self.logger.error(f"Error processing files for {base_name}: {e}")



    def process(self):
        self.logger.info("🚀 Starting data processing")
        self.analyze_zip_encodings()
        self.unzip_and_convert()

        self.merge_files()

        self.logger.info("✅ Processing completed!")

