import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten
from tensorflow.keras.regularizers import l2


def get_model(weights_path=None, compile=True):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(filters=32, kernel_size=5, strides=1, input_shape=(28,28,1), kernel_initializer=tf.keras.initializers.GlorotNormal(), kernel_regularizer=l2(0.0005))) # layer 1
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=5, strides=1, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=False)) # layer 2
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.GlorotNormal(), kernel_regularizer=l2(0.0005)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=False))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=84, use_bias=False, kernel_initializer=tf.keras.initializers.GlorotNormal(), name="WantedLayer"))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dense(units=42, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=False))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dense(units=28, kernel_initializer=tf.keras.initializers.GlorotNormal(), use_bias=False))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(Dense(10))
    model.add(tf.keras.layers.Softmax())

    if compile:
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])
    
    if weights_path is not None:
        model.load_weights(weights_path)

    return model