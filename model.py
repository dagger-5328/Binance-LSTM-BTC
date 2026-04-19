"""
model.py — LSTM + Temporal Attention Architecture
---------------------------------------------------
Components: TemporalAttention layer, FocalLoss, build_model
"""
import tensorflow as tf
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                     LayerNormalization, Layer)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from config import (LSTM_UNITS, DENSE_UNITS, DROPOUT, L2_REG,
                    LEARNING_RATE, TARGET_COLS)

@tf.keras.utils.register_keras_serializable()
class TemporalAttention(Layer):
    """Soft attention: learns which timesteps matter most."""
    def build(self, input_shape):
        self.W = self.add_weight(name='att_w', shape=(input_shape[-1], 1), initializer='glorot_uniform')
        self.b = self.add_weight(name='att_b', shape=(1,), initializer='zeros')
        super().build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(x * weights, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super().get_config()



def build_model(input_shape):
    """LSTM(64)->LSTM(32)->Attention->BatchNorm->Dense->sigmoid(2)"""
    inputs = Input(shape=input_shape)
    x = inputs
    for units in LSTM_UNITS:
        x = LSTM(units, return_sequences=True, kernel_regularizer=l2(L2_REG))(x)
        x = Dropout(DROPOUT)(x)
    x = TemporalAttention(name='attention')(x)
    x = LayerNormalization()(x)
    x = Dense(DENSE_UNITS, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(len(TARGET_COLS), activation='sigmoid', name='predictions')(x)
    model = Model(inputs=inputs, outputs=outputs, name='lstm_attention')
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model
