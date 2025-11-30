# slogan_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import numpy as np
import pickle


class SloganGeneratorModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_sequence_len = 20

    def build_model(self, vocab_size):
        """Создание генеративной модели"""
        self.model = Sequential([
            Embedding(vocab_size, 100, input_length=self.max_sequence_len - 1),
            LSTM(150, return_sequences=True),
            LSTM(100),
            Dense(100, activation='relu'),
            Dense(vocab_size, activation='softmax')
        ])

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train_on_slogans(self, slogans):
        """Обучение на датасете слоганов"""
        # Токенизация
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.utils import to_categorical

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(slogans)

        # Подготовка данных для обучения
        input_sequences = []
        for slogan in slogans:
            token_list = self.tokenizer.texts_to_sequences([slogan])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        # Создание X и y
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre')
        X = input_sequences[:, :-1]
        y = to_categorical(input_sequences[:, -1], num_classes=len(self.tokenizer.word_index) + 1)

        # Обучение
        history = self.model.fit(X, y, epochs=100, verbose=1)
        return history