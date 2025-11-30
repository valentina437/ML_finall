# train_model.py - запустите этот файл один раз
import tensorflow as tf
import numpy as np
import pickle
import os

# Создаем папку для модели
os.makedirs('model', exist_ok=True)

# Простая модель для демонстрации
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(100, 64),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(50, activation='softmax')  # 50 классов
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Сохраняем модель (даже без обучения для демонстрации)
model.save('model/slogan_model.h5')


# Создаем простой токенизатор
class SimpleTokenizer:
    def __init__(self):
        self.word_index = {'start': 1, 'end': 2, 'кофе': 3, 'технологии': 4}


with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(SimpleTokenizer(), f)

print("✅ Демо-модель создана!")