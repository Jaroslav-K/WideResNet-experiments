import numpy as np
from keras import backend as K
import tensorflow as tf

class Backpropagation():
    def __init__(self, model, layer_name, input_data, layer_idx=None, masking=None):
        """
        @params:
            - model: a TensorFlow 2.x Keras Model.
            - layer_name: name of the layer to be backpropagated, can be determined by
              model.layers[layer_idx].name.
            - input_data: input data to be inspected, must be in the proper format
              to be able to be fed into the model.
            - layer_idx: equivalent to layer_name.
            - masking: determine which units in the chosen layer to be backpropagated,
              a numpy array with the same shape as the chosen layer.
        """
        self.model = model
        self.layer_name = layer_name
        self.input_data = input_data
        self.layer = model.get_layer(layer_name)
        
        
        if layer_idx is None:
            for i, layer in enumerate(self.model.layers):
                if layer.name == self.layer_name:
                    self.layer_idx = i
                    break
                    
        if masking is None:
            shape = [1] + list(self.layer.output_shape[1:])
            masking = np.ones(shape, dtype='float32')
        self.masking = masking

    # @tf.function
    def compute(self):
        """
        @returns:
            - output_data: obtained heatmap.
            - func: a reusable function to compute backpropagation in the same setting.
        """
        
        # submodel = tf.keras.models.Model(self.model.input, [self.model.input, self.model.output])
       
        with tf.GradientTape() as tape:
            tape.watch(self.input_data)
            output = self.model(self.input_data, training=False)
            loss = tf.reduce_mean(output * self.masking) #self.layer.output

        gradients = tape.gradient(loss, self.input_data)
        output_data = gradients.numpy()
        output_data = self.filter_gradient(output_data)
        return output_data  

    def filter_gradient(self, x):
        """
        The gradients to be visualized have non-negative values.
        """
        x_abs = np.abs(x)
        x_max = np.amax(x_abs, axis=-1)
        return x_max


class GuidedBackprop(Backpropagation):
    def __init__(self, model, layer_name, input_data, layer_idx=None, masking=None):
        """
        For parameters, please refer to Backpropagation()
        """
        super(GuidedBackprop, self).__init__(
            model, layer_name, input_data, layer_idx, masking
        )

    def compute(self):
        """
        @returns:
          - gradients_input: obtained heatmap.
        """
        forward_values = [self.input_data] + self.feed_forward()
        forward_values_dict = {
            self.model.layers[i].name: forward_values[i]
            for i in range(self.layer_idx + 1)
        }
        gradients = self.masking

        for layer_idx in range(self.layer_idx - 1, -1, -1):
            layer_cur = self.model.layers[layer_idx + 1].output
            layer_prev = self.model.layers[layer_idx].output
            layer_prev_name = self.model.layers[layer_idx].name

            gradients_cur = gradients
            gate_b = (gradients_cur > 0.0) * gradients_cur
            gradients = self.guided_backprop_adjacent(
                layer_cur, layer_prev, forward_values_dict[layer_prev_name], gate_b
            )
            if gradients.min() != gradients.max():
                gradients = self.normalize_gradient(gradients)

        gradients_input = gradients
        gradients_input = self.filter_gradient(gradients_input)
        return gradients_input

    def guided_backprop_adjacent(self, model, layer_cur, layer_prev, values_prev, gate_b):
        # loss = tf.reduce_mean(layer_cur * gate_b)

        # Convert the KerasTensor object to a TensorFlow tensor.
        # layer_prev_output = model.get_layer(name=layer_prev)

        # with tf.GradientTape() as tape:
        #     tape.watch(self.input_data)
        #     output = self.model(self.input_data, training=False)
        #     loss = tf.reduce_mean(output * self.masking) #self.layer.output

        # gradients = tape.gradient(loss, self.input_data)
        # output_data = gradients.numpy()
        # output_data = self.filter_gradient(output_data)
        # return output_data  

        with tf.GradientTape() as tape:
            tape.watch(layer_prev)

            loss = tf.reduce_mean(layer_cur * gate_b)
            # gradients = tape.gradient(loss, layer_prev_output)

        gate_f = tf.cast(values_prev > 0.0, "float32")
        guided_gradients = gradients * gate_f

        func = K.function([self.model.input], [guided_gradients])
        output_data = func([self.input_data])[0]
        # output_data = self.model(self.input_data)
        return output_data

    def feed_forward(self):
        forward_layers = [
            layer.output for layer in self.model.layers[1 : self.layer_idx + 1]
        ]
        func = K.function([self.model.input], forward_layers)
        self.forward_values = func([self.input_data])

        return self.forward_values

    def normalize_gradient(self, img):
        """
        Gradients computed tend to become pretty small, especially after many layers.
        So after each layer, we will multiply them with a constant to keep them in acceptable
        range (if applicable).
        """
        gap = img.max() - img.min()
        if abs(gap) > 1.0:
            return img
        amplitude = 1.0 / gap
        img *= amplitude

        return img


class DeconvNet(GuidedBackprop):
    def __init__(self, model, layer_name, input_data, layer_idx=None, masking=None):
        """
        For parameters, please refer to Backpropagation()
        """
        super(DeconvNet, self).__init__(
            model, layer_name, input_data, layer_idx, masking
        )

    def compute(self):
        """
        @returns:
          - gradients_input: obtained heatmap.
        """
        gradients = self.masking

        for layer_idx in range(self.layer_idx - 1, -1, -1):
            layer_prev = self.model.layers[layer_idx].output
            layer_cur = self.model.layers[layer_idx + 1].output

            forward_values_prev = np.ones(
                [self.input_data.shape[0]]
                + list(self.model.layers[layer_idx].output_shape[1:])
            )

            gradients_cur = gradients
            gate_b = (gradients_cur > 0.0) * gradients_cur
            gradients = self.guided_backprop_adjacent(
                layer_cur, layer_prev, forward_values_prev, gate_b
            )

            if gradients.min() != gradients.max():
                gradients = self.normalize_gradient(gradients)

        gradients_input = gradients
        gradients_input = self.filter_gradient(gradients_input)
        return gradients_input
