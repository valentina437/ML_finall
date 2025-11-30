# create_model.py
import tensorflow as tf
import numpy as np
import pickle
import os

print("🔄 Создаем модель нейронной сети...")

# Создаем папку model если её нет
os.makedirs('model', exist_ok=True)

# Создаем простую нейронную сеть
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,), name='dense_1'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Модель создана!")
print("Архитектура модели:")
model.summary()

# Сохраняем модель
model.save('model/slogan_model.h5')
print("✅ Модель сохранена в model/slogan_model.h5")

# Создаем и сохраняем токенизатор
tokenizer_data = {
    'word_index': {
        'кофе': 1, 'технологии': 2, 'спорт': 3, 'красота': 4,
        'образование': 5, 'еда': 6, 'здоровье': 7, 'путешествия': 8,
        'мода': 9, 'автомобили': 10, 'старт': 11, 'конец': 12
    },
    'config': {
        'num_words': 50,
        'filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        'lower': True
    }
}

with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer_data, f)

print("✅ Токенизатор сохранен в model/tokenizer.pkl")

# Проверяем что файлы созданы
if os.path.exists('model/slogan_model.h5'):
    print("✅ Файл модели существует")
else:
    print("❌ Файл модели НЕ создан")

if os.path.exists('model/tokenizer.pkl'):
    print("✅ Файл токенизатора существует")
else:
    print("❌ Файл токенизатора НЕ создан")

print("🎉 Модель готова к использованию!")