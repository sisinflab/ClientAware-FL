import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Input, Lambda, Add, Multiply, Dropout, BatchNormalization



import tensorflow.keras.backend as K
import tensorflow.keras.initializers as initializers

def Model(sentence_len, char_dim, n_chars, lr):
    model = tf.keras.Sequential()
    model.add(Embedding(n_chars, char_dim, input_length=sentence_len))
    # model.add(tf.keras.layers.LSTM(256, input_shape=(sentence_len, char_dim), return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256))
    model.add(tf.keras.layers.Dense(n_chars, activation='softmax'))
    model.compile(optimizer=optimizers.SGD(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# def Highway(value, activation, gate_bias=-3):
#     dim = K.int_shape(value)[-1]
#     gate_bias_initializer = initializers.Constant(gate_bias)
#
#     gate = Dense(units=dim, bias_initializer=gate_bias_initializer, activation='sigmoid')(value)
#     negated_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(gate)
#     transformed = Dense(units=dim, activation=activation)(value)
#     transformed_gated = Multiply()([gate, transformed])
#     identity_gated = Multiply()([negated_gate, value])
#     value = Add()([transformed_gated, identity_gated])
#     return value
#
# def Model(sentence_len, char_dim, n_chars, lr):
#     inputs = Input(shape=(sentence_len,))
#     x = Embedding(n_chars, char_dim, input_length=sentence_len)(inputs)
#
#
#     x = Conv1D(filters=25, kernel_size=1, activation='tanh')(x)
#     x = MaxPooling1D()(x)
#     x = Conv1D(filters=50, kernel_size=2, activation='tanh')(x)
#     x = MaxPooling1D()(x)
#     x = Conv1D(filters=75, kernel_size=3, activation='tanh')(x)
#     x = MaxPooling1D()(x)
#     x = Conv1D(filters=100, kernel_size=4, activation='tanh')(x)
#     x = MaxPooling1D()(x)
#
#     x = Highway(x, activation='relu')
#     x = Highway(x, activation='relu')
#
#     x = LSTM(300, return_sequences=True)(x)
#     x = LSTM(300)(x)
#     x = Dropout(0.5)(x)
#     x = Dense(n_chars, activation='softmax')(x)
#
#     model = tf.keras.models.Model(inputs=inputs, outputs=x)
#     model.compile(optimizer=optimizers.SGD(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


# def Model(sentence_len, char_dim, n_chars, lr):
#     # conv_layers = [[256,7],[256,5],[256,3],[256,1]]
#     filter_widths = [1, 2, 3, 4, 5, 6]
#
#     model = tf.keras.Sequential()
#     # model.add(Input(shape=(sentence_len,)))
#     model.add(Embedding(n_chars, char_dim, input_length=sentence_len))
#
#     # for filter_width in filter_widths:
#     model.add(Conv1D(filters=25, kernel_size=1, activation='tanh'))
#     model.add(MaxPooling1D())
#
#     model.add(LSTM(256, return_sequences=True))
#     model.add(LSTM(256))
#     model.add(Dense(n_chars, activation='softmax'))
#     model.compile(optimizer=optimizers.SGD(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model



# model = Model(80,15,58,0.001)
# print(model.summary())
