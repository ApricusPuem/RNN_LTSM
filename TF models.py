# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import random

# Генерация синтетических данных
noice = [random.randint(1, 10) for i in range(10000)]
sin = [np.sin(i) for i in range(10000)]
linear = [i / 10 for i in range(10000)]
df = pd.DataFrame({'noice': noice,
                   'sin': sin,
                   'linear': linear,
                   'sum': [sum([i, j, k]) for i, j, k in zip(noice, sin, linear)]})
df.to_csv('out.csv', index=False)  

column_indices = {name: i for i, name in enumerate(df.columns)}

# Разбиваем датасет на тренировочный, валидирующих и тестирующий
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

# Нормализация данных
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')


# Класс для создания окон данных
class WindowGenerator():
    # Ширина окна, сдвиг и метки
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        # Хранение сырых данных
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Обработка столбцов меток и присвоение индексов
        self.label_columns = label_columns
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Обработка параметров окна данных
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


    # Выделение нужного количества данных созармерно окну данных из поданных данных
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Срезы не представляют информацию о shape'е, поэтому необходимо
        # их задавать самостоятельно. Так `tf.data.Datasets` проще исследовать
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    # Вывод графиков для заданного окна
    def plot(self, model=None, plot_col='noice', model_name=None, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        plt.title(f'Plot of: {plot_col}\nInput: {model_name}\nOutputs: {self.label_columns if self.label_columns is not None else "All columns"}')
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                try:
                    plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                                marker='X', edgecolors='k', label='Predictions',
                                c='#ff7f0e', s=64)
                except Exception:
                    plt.scatter(self.label_indices, np.transpose(predictions[n, :]),
                                marker='X', edgecolors='k', label='Predictions',
                                c='#ff7f0e', s=64)


            if n == 0:
                plt.legend()

        plt.xlabel("Values' indicies")


    # Создание tf.data.Datasets из заданного набора данных
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)
        return ds

    # Инициализация заданных датасетов
    @property
    def train(self):
      return self.make_dataset(self.train_df)

    @property
    def val(self):
      return self.make_dataset(self.val_df)

    @property
    def test(self):
      return self.make_dataset(self.test_df)

    # Создание и обработка части используемого датасета для демонстрации
    @property
    def example(self):
      """Get and cache an example batch of `inputs, labels` for plotting."""
      result = getattr(self, '_example', None)
      if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
      return result

    @example.setter
    def example(self, value):
        self._example = value


# Функция для компиляции и обучения модели
MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

val_performance = {}
performance = {}

# Инициализация моделей с различным входом и соответствующих окон

# Модель шума
lstm_model_noice = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

# Ширина: 150, сдвиг: 1, метки: 150, столбцы меток: шум
wide_window_noice = WindowGenerator(
    input_width=150, label_width=150, shift=1,
    label_columns=['noice'])

# Модель синуса
lstm_model_sin = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

# Ширина: 150, сдвиг: 1, метки: 150, столбцы меток: синус
wide_window_sin = WindowGenerator(
    input_width=150, label_width=150, shift=1,
    label_columns=['sin'])

# Линейная модель
lstm_model_linear = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

# Ширина: 150, сдвиг: 1, метки: 150, столбцы меток: линейная функция
wide_window_linear = WindowGenerator(
    input_width=150, label_width=150, shift=1,
    label_columns=['linear'])

# Суммарная модель
lstm_model_sum = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

# Ширина: 150, сдвиг: 1, метки: 150, столбцы меток: сумма
wide_window_sum = WindowGenerator(
    input_width=150, label_width=150, shift=1,
    label_columns=['sum'])

# Множественная модель (на вход подаются шум, синус, линейная функция и их сумма)
lstm_model_multiple = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=4)
])

# Ширина: 150, сдвиг: 1, метки: 150, столбцы меток: все
wide_window_multiple = WindowGenerator(
    input_width=150, label_width=150, shift=1,
    label_columns=None)


# Компиляция и обучение моделей
history_noice = compile_and_fit(lstm_model_noice, wide_window_noice)
history_sin = compile_and_fit(lstm_model_sin, wide_window_sin)
history_linear = compile_and_fit(lstm_model_linear, wide_window_linear)
history_sum = compile_and_fit(lstm_model_sum, wide_window_sum)
history_multiple = compile_and_fit(lstm_model_multiple, wide_window_multiple)

# Вывод графиков всех моделей

# Вход: шум, вывод: шум, выход: шум
wide_window_noice.plot(lstm_model_noice, 'noice', 'lstm_model_noice')

# Вход: синус, вывод: синус, выход: синус
wide_window_sin.plot(lstm_model_sin, 'sin', 'lstm_model_sin')

# Вход: линейная функция, вывод: линейная функция, выход: линейная функция
wide_window_linear.plot(lstm_model_linear, 'linear', 'lstm_model_linear')

# Вход: сумма, вывод: сумма, выход: сумма
wide_window_sum.plot(lstm_model_sum, 'sum', 'lstm_model_sum')

# Вход: все столбцы, вывод: шум, выход: все столбцы
wide_window_multiple.plot(lstm_model_multiple, 'noice', 'lstm_model_multiple')

# Вход: все столбцы, вывод: синус, выход: все столбцы
wide_window_multiple.plot(lstm_model_multiple, 'sin', 'lstm_model_multiple')

# Вход: все столбцы, вывод: линейная функция, выход: все столбцы
wide_window_multiple.plot(lstm_model_multiple, 'linear', 'lstm_model_multiple')

# Вход: все столбцы, вывод: сумма, выход: все столбцы
wide_window_multiple.plot(lstm_model_multiple, 'sum', 'lstm_model_multiple')

# Вход: все столбцы, вывод: линейная функция, выход: линейная функция
wide_window_linear.plot(lstm_model_multiple, 'linear', 'lstm_model_multiple')

# Вход: линейная функция, вывод: синус, выход: все столбцы
wide_window_multiple.plot(lstm_model_linear, 'sin', 'lstm_model_linear')


plt.show()