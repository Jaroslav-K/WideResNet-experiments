import tensorflow as tf

augmentation_cifar = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(40, 40, 3)),
    tf.keras.layers.RandomFlip(mode='horizontal'),
    tf.keras.layers.RandomCrop(32, 32)
])


def prepare_dataset_pipeline(x, y, batch_size=128, shuffle=False, augmentation=None, num_parallel_calls=tf.data.AUTOTUNE, prefetch_buffer_size=tf.data.AUTOTUNE):
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        ds = ds.shuffle(len(y))

    ds = ds.batch(batch_size)

    if augmentation is not None:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=num_parallel_calls)

    ds = ds.prefetch(prefetch_buffer_size)

    return ds