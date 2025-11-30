# model.py - настоящая нейросеть для классификации
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import numpy as np
import json


class SentimentModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def build_model(self, vocab_size, max_length):
        """Создание модели для классификации тональности"""
        self.model = Sequential([
            Embedding(vocab_size, 100, input_length=max_length),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')  # 3 класса: позитив, негатив, нейтраль
        ])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, texts, labels):
        """Обучение модели на реальных данных"""
        # Токенизация текстов
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=100)

        # Обучение модели
        history = self.model.fit(X, labels, epochs=10, validation_split=0.2)
        return history