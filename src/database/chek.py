# check_db.py

import sqlite3
from pathlib import Path

db_path = Path('data/sqlite_data/experiments.db')
print(f'БД существует: {db_path.exists()}')
print(f'Размер: {db_path.stat().st_size if db_path.exists() else 0} байт')

if db_path.exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Список таблиц
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f'Таблицы: {[t[0] for t in tables]}')
    
    # Количество записей в predictions
    cursor.execute('SELECT COUNT(*) FROM predictions;')
    count = cursor.fetchone()[0]
    print(f'Записей в predictions: {count}')
    
    # Последние 5 записей
    cursor.execute('''
        SELECT id, model_name, num_objects, inference_time_ms, success, created_at 
        FROM predictions 
        ORDER BY id DESC 
        LIMIT 5;
    ''')
    rows = cursor.fetchall()
    print('\nПоследние записи:')
    for row in rows:
        print(f'  {row}')
    
    conn.close()
else:
    print('БД не найдена')