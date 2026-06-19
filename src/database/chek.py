# src/migrate/check.py
import sys
import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SQLITE_PATH = PROJECT_ROOT / "data" / "sqlite_data" / "experiments.db"

def check_database():
    print(f"Database: {SQLITE_PATH}")
    
    if not SQLITE_PATH.exists():
        print("Database not found")
        return
    
    try:
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"Tables: {[t[0] for t in tables]}")
            
            for table_name in [t[0] for t in tables]:
                df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
                
                if not df.empty:
                    print(f"\nTable: {table_name}")
                    print(f"Records: {len(pd.read_sql_query(f'SELECT * FROM {table_name}', conn))}")
                    print(f"Columns: {list(df.columns)}")
                    print(df.to_string(index=False))
                else:
                    print(f"\nTable: {table_name} (empty)")
            
            print("\nModel Statistics")
            df_exp = pd.read_sql_query("""
                SELECT 
                    model_name,
                    model_type,
                    COUNT(*) as experiments,
                    AVG(mAP50) as avg_mAP50,
                    AVG(mAP50_95) as avg_mAP50_95,
                    AVG(precision) as avg_precision,
                    AVG(recall) as avg_recall
                FROM experiments
                GROUP BY model_name, model_type
                ORDER BY avg_mAP50 DESC
            """, conn)
            
            if not df_exp.empty:
                print(df_exp.to_string(index=False))
            
            print("Check completed")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database()