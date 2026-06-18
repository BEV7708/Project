"""
Smoke test для проверки работоспособности API
"""

import os
import sys
import json
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Проверка здоровья сервиса"""
    print("\n[1] Проверка здоровья сервиса...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Сервис работает: {data['status']}")
            print(f"  📊 Загружено моделей: {data['models_loaded']}")
            return True
        else:
            print(f"  ❌ Ошибка: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Не удалось подключиться: {e}")
        return False

def test_predict():
    """Тест эндпоинта /predict"""
    print("\n[2] Тест эндпоинта /predict...")
    
    # Ищем тестовое изображение
    test_images = list(Path("data/data/test/images").glob("*.jpg")) + \
                  list(Path("data/data/test/images").glob("*.png"))
    
    if not test_images:
        print("  ⚠️ Нет тестовых изображений")
        return False
    
    img_path = test_images[0]
    print(f"  📷 Изображение: {img_path}")
    
    try:
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                params={"model_name": "yolo_n", "device": "cpu"},
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Успешно")
            print(f"  📊 Найдено объектов: {data['num_objects']}")
            print(f"  📊 Уверенность: {data['confidence']:.3f}")
            print(f"  ⏱️ Время инференса: {data['inference_time']:.3f}с")
            return True
        else:
            print(f"  ❌ Ошибка: {response.status_code}")
            print(f"  {response.text}")
            return False
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False

def test_stats():
    """Тест эндпоинта /stats"""
    print("\n[3] Тест эндпоинта /stats...")
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Успешно")
            if data.get('models'):
                print(f"  📊 Моделей в статистике: {len(data['models'])}")
                for model in data['models'][:3]:
                    print(f"    - {model['model_name']}: {model['total_predictions']} запросов")
            return True
        else:
            print(f"  ❌ Ошибка: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False

def test_history():
    """Тест эндпоинта /history"""
    print("\n[4] Тест эндпоинта /history...")
    try:
        response = requests.get(f"{API_URL}/history", params={"limit": 5}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Успешно")
            print(f"  📊 Записей: {data['total']}")
            return True
        else:
            print(f"  ❌ Ошибка: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False

def test_compare():
    """Тест эндпоинта /compare"""
    print("\n[5] Тест эндпоинта /compare...")
    test_images = list(Path("data/data/test/images").glob("*.jpg")) + \
                  list(Path("data/data/test/images").glob("*.png"))
    
    if not test_images:
        print("  ⚠️ Нет тестовых изображений")
        return False
    
    img_name = test_images[0].name
    print(f"  📷 Изображение: {img_name}")
    
    try:
        response = requests.get(
            f"{API_URL}/compare",
            params={"image_file": img_name, "model_names": "yolo_n,yolo_s,rtdetr"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Успешно")
            if data.get('results'):
                print(f"  📊 Сравнено моделей: {len(data['results'])}")
                for model_name, result in data['results'].items():
                    if 'num_objects' in result:
                        print(f"    - {model_name}: {result['num_objects']} объектов, "
                              f"{result['inference_time']:.3f}с")
            return True
        else:
            print(f"  ❌ Ошибка: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False

def main():
    print("=" * 60)
    print("SMOKE TEST - Проверка API сервиса")
    print("=" * 60)
    
    results = {
        "health": test_health(),
        "predict": test_predict(),
        "stats": test_stats(),
        "history": test_history(),
        "compare": test_compare()
    }
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print("-" * 60)
    print(f"  ИТОГО: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("  ✅ Все тесты пройдены успешно!")
    else:
        print("  ⚠️ Некоторые тесты не пройдены")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    sys.exit(0 if main() else 1)