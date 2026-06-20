import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

try:
    from src.database.database import get_db
    db = get_db()
    print("БД подключена")
    
    # Добавить тестовые данные
    for i in range(5):
        db.log_prediction({
            'model_name': 'yolo_n',
            'image_path': f'test_{i+1}.jpg',
            'num_objects': i + 1,
            'confidence': 0.9,
            'inference_time_ms': 150.0 + i * 10,
            'device': 'cpu',
            'success': True
        })
        print(f'Добавлена запись {i+1}')
    
    print('\nПроверка:')
    df = db.get_predictions(limit=10)
    print(df)
    
except Exception as e:
    print(f'Ошибка: {e}')