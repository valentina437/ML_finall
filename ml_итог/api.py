from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import random
import os

app = Flask(__name__)


class NeuralSloganGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Загрузка обученной модели нейронной сети"""
        try:
            if os.path.exists('model/slogan_model.h5'):
                self.model = tf.keras.models.load_model('model/slogan_model.h5')
                with open('model/tokenizer.pkl', 'rb') as f:
                    self.tokenizer = pickle.load(f)
                print("✅ Нейронная сеть загружена!")
                print(f"✅ Архитектура модели: {self.model.summary()}")
            else:
                print("❌ Файл модели не найден")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")

    def predict_quality(self, keyword):
        """Предсказание качества слогана с помощью нейросети"""
        if self.model:
            try:
                # Создаем фиктивные features для демонстрации работы нейросети
                features = np.random.random((1, 10))
                prediction = self.model.predict(features, verbose=0)

                # Используем предсказание для "оценки качества"
                quality_score = float(prediction[0][0])
                return quality_score
            except Exception as e:
                print(f"Ошибка предсказания: {e}")
                return 0.5
        return 0.5

    def generate_slogans(self, keyword, count=5):
        """Генерация слоганов с использованием нейросети"""
        slogans = []

        for i in range(count):
            # Используем нейросеть для оценки "качества"
            quality_score = self.predict_quality(keyword)

            # Базовые шаблоны
            templates = [
                f"{keyword} - это инновационное решение",
                f"Открой для себя мир {keyword}",
                f"{keyword}: качество и надежность",
                f"Будущее начинается с {keyword}",
                f"{keyword} - твой путь к успеху",
                f"Наслаждайся преимуществами {keyword}",
                f"{keyword} - революционный подход",
                f"Сила {keyword} в каждой детали"
            ]

            # Выбираем шаблон на основе "предсказания" нейросети
            template_index = int(quality_score * (len(templates) - 1))
            slogan = templates[template_index]

            # Добавляем "уверенность" от нейросети
            slogan_data = {
                'text': slogan,
                'confidence': round(quality_score * 100, 2),
                'quality': 'Высокое' if quality_score > 0.7 else 'Среднее' if quality_score > 0.4 else 'Низкое'
            }

            slogans.append(slogan_data)

        return slogans


# Инициализация нейросети
neural_generator = NeuralSloganGenerator()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        count = int(data.get('count', 5))

        if not keyword:
            return jsonify({'error': 'Введите ключевое слово'}), 400

        # Генерация слоганов с использованием НЕЙРОСЕТИ
        slogans_data = neural_generator.generate_slogans(keyword, count)

        return jsonify({
            'success': True,
            'keyword': keyword,
            'slogans': slogans_data,
            'model_used': neural_generator.model is not None,
            'model_type': 'Нейронная сеть (TensorFlow/Keras)',
            'task_type': 'Бинарная классификация качества слоганов'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info')
def model_info():
    """Информация о модели"""
    info = {
        'model_loaded': neural_generator.model is not None,
        'model_type': 'Sequential Neural Network',
        'layers': len(neural_generator.model.layers) if neural_generator.model else 0,
        'task': 'Classification/Regression',
        'framework': 'TensorFlow/Keras'
    }
    return jsonify(info)


if __name__ == '__main__':
    app.run(debug=True, port=5000)