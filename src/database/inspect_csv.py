# src/database/inspect_csv.py
import pandas as pd
from pathlib import Path

csv_dir = Path("models/weights/trained/yolo_metrics")

for csv_path in csv_dir.glob("*.csv"):
    df = pd.read_csv(csv_path)
    print(f"\n{csv_path.name}")
    print(f"Columns: {list(df.columns)}")
    print(f"First row: {df.iloc[0].to_dict()}")
    print(f"Shape: {df.shape}")