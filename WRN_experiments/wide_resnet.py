import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    SpatialDropout2D,
)


class WideResnetBuilder:
    use_bias = False
    weight_decay = 0.0005
    dropout_probability = 0.0
    weight_init = "he_normal"
    channel_axis = -1

    def __init__(
        self,
        use_bias: bool = False,
        weight_decay: float = 0.0005,
        dropout_probability: float = 0.0,
        weight_init: str = "he_normal",
        channel_axis: int = -1,
    ):
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        self.dropout_probability = dropout_probability
        self.weight_init = weight_init
        self.channel_axis = channel_axis

    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):
            # format of conv_params:
            #               [ [nb_col="kernel width", nb_row="kernel height",
            #               subsample="(stride_vertical,stride_horizontal)",
            #               border_mode="same" or "valid"] ]
            # B(3,3): orignal <<basic>> block
            conv_params = [[3, 3, stride, "same"], [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane

            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self.channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(axis=self.channel_axis)(net)
                        convs = Activation("relu")(convs)
                    convs = Conv2D(
                        n_bottleneck_plane,
                        (v[0], v[1]),
                        strides=v[2],
                        padding=v[3],
                        kernel_initializer=self.weight_init,
                        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                        use_bias=self.use_bias,
                    )(convs)
                else:
                    convs = BatchNormalization(axis=self.channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self.dropout_probability > 0:
                        convs = SpatialDropout2D(self.dropout_probability)(convs)
                    convs = Conv2D(
                        n_bottleneck_plane,
                        (v[0], v[1]),
                        strides=v[2],
                        padding=v[3],
                        kernel_initializer=self.weight_init,
                        kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                        use_bias=self.use_bias,
                    )(convs)

            # Shortcut Conntection: identity function or 1x1 convolutional
            #  (depends on difference between input & output shape - this
            #   corresponds to whether we are using the first block in each
            #   group; see _layer() ).
            if n_input_plane != n_output_plane:
                shortcut = Conv2D(
                    n_output_plane,
                    (1, 1),
                    strides=stride,
                    padding="same",
                    kernel_initializer=self.weight_init,
                    kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                    use_bias=self.use_bias,
                )(net)
            else:
                shortcut = net

            return Add()([convs, shortcut])

        return f

    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

    def create_model(self, input_shape=(32, 32, 3), depth=40, k=4):
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6

        inputs = Input(shape=input_shape)

        n_stages = [16, 16 * k, 32 * k, 64 * k]

        conv1 = Conv2D(
            n_stages[0],
            (3, 3),
            strides=1,
            padding="same",
            kernel_initializer=self.weight_init,
            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
            use_bias=self.use_bias,
        )(
            inputs
        )  # "One conv at the beginning (spatial size: 32x32)"

        # Add wide residual blocks
        block_fn = self._wide_basic
        conv2 = self._layer(
            block_fn,
            n_input_plane=n_stages[0],
            n_output_plane=n_stages[1],
            count=n,
            stride=(1, 1),
        )(
            conv1
        )  # "Stage 1 (spatial size: 32x32)"
        conv3 = self._layer(
            block_fn,
            n_input_plane=n_stages[1],
            n_output_plane=n_stages[2],
            count=n,
            stride=(2, 2),
        )(
            conv2
        )  # "Stage 2 (spatial size: 16x16)"
        conv4 = self._layer(
            block_fn,
            n_input_plane=n_stages[2],
            n_output_plane=n_stages[3],
            count=n,
            stride=(2, 2),
        )(
            conv3
        )  # "Stage 3 (spatial size: 8x8)"

        batch_norm = BatchNormalization(axis=self.channel_axis)(conv4)
        output = Activation("relu")(batch_norm)

        model = tf.keras.models.Model(inputs=inputs, outputs=output)
        return model
