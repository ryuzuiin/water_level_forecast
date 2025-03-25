import os
import sys
import traceback
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from water_level_forecast.japanese_data_processor import JapaneseDataProcessor
from water_level_forecast.data_cleaning import DataCleaner
from water_level_forecast.data_filter import DataFilter
from water_level_forecast.anomaly_handler import AnomalyHandler
from water_level_forecast.ml_data_preparer import MLInputPreparation
from water_level_forecast.train_lightgbm import LightGBMTrainer
from water_level_forecast.lightgbm_predictor import LightGBMPredictor
from water_level_forecast.feature_library import DynamicFeatureLibrary

def process_japanese_data(input_dir, output_dir):
    print("Processing Japanese water level data...")
    processor = JapaneseDataProcessor(input_dir, output_dir, max_workers=4)
    processor.process()

def process_filtered_data(output_dir):
    print("Processing filtered data...")
    data_filter = DataFilter(output_dir)
    data_filter.process_all_files()

def clean_data(output_dir):
    print("Cleaning data...")
    data_cleaner = DataCleaner(output_dir)
    data_cleaner.clean_all_files()

def handle_anomalies(output_dir):
    print("Handling anomalies...")
    data_handler = AnomalyHandler(output_dir)
    data_handler.process_all_files()

def prepare_ml_data(output_dir):
    print("Preparing machine learning data...")
    ml_preparer = MLInputPreparation(output_dir)
    ml_preparer.process_all()


def prepare_dynamic_features(output_dir, config_file=None):
    print("Preparing dynamic features for all headworks...")
    feature_library = DynamicFeatureLibrary(output_dir, config_file)
    feature_library.prepare_all_headworks()
    print("✅ Dynamic feature preparation completed for all headworks.")

def train_model(output_dir, target_headworks, with_rainfall: bool):
    print(f"🚀 Training LightGBM models for {target_headworks}...")
    trainer = LightGBMTrainer(output_dir, target_headworks, with_rainfall)
    trainer.run()
    print(f"✅ Model training completed for {target_headworks}.")

def predict_model(output_dir, target_headworks, with_rainfall: bool):
    print(f"🔮 Running predictions for {target_headworks}...")
    predictor = LightGBMPredictor(output_dir, target_headworks, with_rainfall)
    predictor.run()
    print(f"✅ Predictions saved for {target_headworks}.")

def main():
    parser = argparse.ArgumentParser(description="Water Level Data Processing, Model Training, and Prediction")
    parser.add_argument("--process", choices=["all", "japanese", "filter", "clean", "anomaly", "ml", 
                                             "dynamic_features", "train", "predict"], 
                        required=True, help="Specify the process step to execute")
    parser.add_argument("--target", type=str, help="Target headworks to train or predict on")
    parser.add_argument("--with_rainfall", action="store_true", help="Use dataset with rainfall if set")
    parser.add_argument("--config", type=str, help="Path to feature configuration file (optional)")

    args = parser.parse_args()


    # NOTE: Modify this as needed
    input_dir = "data/データファイル"
    output_dir = os.path.join(input_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    try:
        if args.process == "japanese":
            process_japanese_data(input_dir, output_dir)

        elif args.process == "filter":
            process_filtered_data(output_dir)

        elif args.process == "clean":
            clean_data(output_dir)

        elif args.process == "anomaly":
            handle_anomalies(output_dir)

        elif args.process == "ml":
            prepare_ml_data(output_dir)

        elif args.process == "dynamic_features":
            prepare_dynamic_features(output_dir, args.config)

        elif args.process == "train":
            if not args.target:
                raise ValueError("Target headworks must be specified for training.")
            train_model(output_dir, args.target, args.with_rainfall)

        elif args.process == "predict":
            if not args.target:
                raise ValueError("Target headworks must be specified for prediction.")
            predict_model(output_dir, args.target, args.with_rainfall)

        elif args.process == "all":
            if not args.target:
                raise ValueError("❗ --target must be specified for 'all' process (for train & predict).")
            

            process_japanese_data(input_dir, output_dir)
            process_filtered_data(output_dir)
            clean_data(output_dir)
            handle_anomalies(output_dir)
            prepare_ml_data(output_dir)
            prepare_dynamic_features(output_dir)
            train_model(output_dir, args.target, args.with_rainfall)
            predict_model(output_dir, args.target, args.with_rainfall)

        print("\n✅ Selected processing step completed!")

    except Exception as e:
        print(f"❌ Error in processing: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()