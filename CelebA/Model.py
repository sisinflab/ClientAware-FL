import tensorflow as tf
from tensorflow.keras import optimizers
from keras.initializers import glorot_normal

def Model(lr, image_size):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(image_size, image_size, 3)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=5, padding='same', kernel_initializer=glorot_normal()))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer=glorot_normal()))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=glorot_normal()))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=glorot_normal()))

    model.compile(optimizer=optimizers.SGD(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

