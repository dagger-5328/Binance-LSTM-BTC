"""
LSTM Model Architecture with Temporal Attention
===============================================
Custom TemporalAttention layer for focusing on relevant timesteps.
Multi-task output for 3d and 7d horizons.
"""

import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

@keras.saving.register_keras_serializable(package="Custom")
class TemporalAttention(tf.keras.layers.Layer):
    """
    Custom attention layer that learns to weight timesteps in LSTM sequences.
    Focuses on the most relevant historical patterns for prediction.
    """

    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, features)
        self.W = self.add_weight(name="attention_weight",
                                shape=(input_shape[-1], 1),
                                initializer="glorot_uniform",
                                trainable=True)
        self.b = self.add_weight(name="attention_bias",
                                shape=(input_shape[1], 1),
                                initializer="zeros",
                                trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, timesteps, features)
        # Compute attention scores
        e = tf.matmul(inputs, self.W) + self.b  # (batch_size, timesteps, 1)
        e = tf.squeeze(e, axis=-1)  # (batch_size, timesteps)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(e, axis=-1)  # (batch_size, timesteps)

        # Apply attention weights to input
        weighted_input = inputs * tf.expand_dims(attention_weights, axis=-1)  # (batch_size, timesteps, features)

        # Sum across timesteps
        context_vector = tf.reduce_sum(weighted_input, axis=1)  # (batch_size, features)

        return context_vector

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        return config


def build_model(input_shape, horizons=2):
    """
    Build multi-horizon LSTM model with temporal attention.

    Args:
        input_shape: Tuple (timesteps, features)
        horizons: Number of prediction horizons (default 2 for 3d/7d)

    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)

    # First LSTM layer with return sequences for attention
    lstm1 = LSTM(32, return_sequences=True, kernel_regularizer=l2(0.002))(inputs)
    dropout1 = Dropout(0.4)(lstm1)

    # Temporal attention mechanism
    attention = TemporalAttention()(dropout1)
    norm = LayerNormalization()(attention)

    # Dense layers
    dense1 = Dense(16, activation="relu", kernel_regularizer=l2(0.002))(norm)
    dropout2 = Dropout(0.2)(dense1)

    # Multi-task output (one for each horizon)
    outputs = Dense(horizons, activation="sigmoid")(dropout2)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0008),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
