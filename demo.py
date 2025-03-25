import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import japanize_matplotlib
import time
from pathlib import Path

class WaterLevelDemo:
    def __init__(self, processed_dir: str, target_headworks: str):
        self.processed_dir = Path(processed_dir)
        self.target_headworks = target_headworks
        self.ml_output_dir = self.processed_dir / "ml_output"
        self.demo_output_dir = self.processed_dir / 'demo_output'
        self.output_dir = self.demo_output_dir / 'result'
        self.ml_output_dir.mkdir(parents=True, exist_ok=True)
        self.demo_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.pred_file = self.ml_output_dir / f"{self.target_headworks}_predictions.csv"
        try:
            self.pred_df = pd.read_csv(self.pred_file, encoding="utf-8-sig", parse_dates=["datetime"])
        except Exception as e:
            print(f"❌ 予測データのロードに失敗: {str(e)}")
            self.pred_df = None
            return False

        self.pred_df.sort_values("datetime", inplace=True)
        return True

    def get_prediction(self, base_time: str):
        start_time = time.time()
        base_time = pd.to_datetime(base_time)

        horizons = [10, 20, 30, 40, 50, 60]
        future_times = [base_time + timedelta(minutes=h) for h in horizons]

        actual_levels, predicted_levels = [], []
        actual_rates, predicted_rates = [], []

        water_col = f"{self.target_headworks} 河川水位"

        row_base = self.pred_df[self.pred_df["datetime"] == base_time]
        if row_base.empty:
            print(f"❌ 指定した時刻 {base_time} のデータが見つかりません。")
            return
        actual_now = row_base[water_col].values[0]

        for h, future_time in zip(horizons, future_times):
            row_future = self.pred_df[self.pred_df["datetime"] == future_time]

            pred_col = f"{self.target_headworks}_{h}分後水位"
            future_pred = row_base[pred_col].values[0] if pred_col in row_base.columns else np.nan
            predicted_levels.append(round(future_pred, 2) if not np.isnan(future_pred) else np.nan)

            actual_future = row_future[water_col].values[0] if not row_future.empty else np.nan
            actual_levels.append(round(actual_future, 2) if not np.isnan(actual_future) else np.nan)

            rate_pred_col = f"{h}分後変動率_pred"
            predicted_rate = row_base[rate_pred_col].values[0] * 100 if rate_pred_col in row_base.columns else np.nan
            predicted_rates.append(round(predicted_rate, 2) if not np.isnan(predicted_rate) else "N/A")

            rate_actual_col = f"{h}分後変動率_actual"
            actual_rate = row_future[rate_actual_col].values[0] * 100 if not row_future.empty and rate_actual_col in row_future.columns else np.nan
            actual_rates.append(round(actual_rate, 2) if not np.isnan(actual_rate) else "N/A")

        self.plot_actual_vs_predict(future_times, actual_levels, predicted_levels, base_time)

        result_table = pd.DataFrame({
            "時間": [t.strftime('%Y-%m-%d %H:%M:%S') for t in future_times],
            "予測の水位 (m)": predicted_levels,
            "実測の水位 (m)": actual_levels,
            "予測の水位変動率 (%)": predicted_rates,
            "実測の水位変動率 (%)": actual_rates
        })

        table_file = self.output_dir / f"{self.target_headworks}_予測結果.csv"
        result_table.to_csv(table_file, encoding="utf-8-sig", index=False)

        messages = {
            "基準時刻": base_time.strftime('%Y-%m-%d %H:%M:%S'),
            "現在の水位": round(actual_now, 2),
            "予測の一時間後の水位": round(predicted_levels[-1], 2) if not np.isnan(predicted_levels[-1]) else "N/A",
            "処理時間": round(time.time() - start_time, 4)
        }

        json_file = self.output_dir / f"{self.target_headworks}_予測情報.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

        print(f"\n=== {messages['基準時刻']} の予測 ===")
        print(f"現在の水位: {messages['現在の水位']} m")
        print(f"予測の一時間後の水位: {messages['予測の一時間後の水位']} m")
        print(f"処理時間: {messages['処理時間']} 秒")

        self.display_table(result_table)

    def plot_actual_vs_predict(self, future_times, actual_levels, predicted_levels, base_time):
        plt.figure(figsize=(10, 5))
        plt.plot(future_times, actual_levels, marker="o", label="実測(未来データなし→空)")
        plt.plot(future_times, predicted_levels, marker="s", linestyle="--", label="予測")

        plt.axhline(y=actual_levels[0], color='gray', linestyle='dotted', label="現在の水位")

        plt.xlabel("時間")
        plt.ylabel("水位(m)")
        plt.title(f"{self.target_headworks} 水位予測 (基準時刻: {base_time.strftime('%Y-%m-%d %H:%M:%S')})")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_file = self.output_dir / f"{self.target_headworks}_予測グラフ.png"
        plt.savefig(plot_file)
        plt.show()

    def display_table(self, df):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("tight")
        ax.axis("off")
        ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")

        table_file = self.output_dir / f"{self.target_headworks}_予測テーブル.png"
        plt.savefig(table_file, bbox_inches="tight")
        plt.show()
