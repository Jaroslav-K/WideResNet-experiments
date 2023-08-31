import tensorflow as tf
from wide_resnet import WideResnetBuilder

def get_model(depth, k, n_features=256, output_units=10, use_softmax=False, compile=True):
    builder = WideResnetBuilder()
    model = builder.create_model(input_shape=(32,32,3), depth=depth, k=k)
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.Dense(units=n_features, use_bias=False, name="ClassifierInput")(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(units=output_units, use_bias=False)(x)

    if use_softmax:
        x = tf.keras.layers.Softmax()(x)

    model = tf.keras.models.Model(model.input, x)

    if compile:
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(0.1, nesterov=True, momentum=0.9),
            metrics=['accuracy'])

    return model