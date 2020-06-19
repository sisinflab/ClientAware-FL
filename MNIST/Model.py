import tensorflow as tf
from tensorflow.keras import optimizers
from keras.initializers import glorot_normal

def Model(lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=5, padding='same', input_shape=(28, 28, 1), kernel_initializer=glorot_normal()),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer=glorot_normal()),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer=glorot_normal()),
        tf.keras.layers.Dense(10, activation='softmax',kernel_initializer=glorot_normal()) # units = 10 if MNIST, units = 62 if FEMNIST
    ])
    #model = tf.keras.utils.multi_gpu_model(model, gpus=4)
    model.compile(optimizer=optimizers.SGD(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

