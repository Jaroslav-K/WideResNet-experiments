import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

@tf.RegisterGradient("DeconvReLU")
def DeconvReLU(op, grad):
    return tf.cast(grad > 0, tf.float32) * grad

@tf.RegisterGradient("GuidedReLU")
def GuidedReLU(op, grad):
    return tf.cast(op.outputs[0] > 0, tf.float32) * tf.cast(grad > 0, tf.float32) * grad

@tf.custom_gradient
def DeconvReLU(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * dy
    return tf.nn.relu(x), grad

@tf.custom_gradient
def GuidedReLU(x):
    def grad(dy):
        return tf.cast(x > 0, "float32") * tf.cast(dy > 0, "float32") * dy
    return tf.nn.relu(x), grad

class Interpreter:
    
    model = None

    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name

    def __clone_model(self):
        cloned_model = tf.keras.models.clone_model(self.model)
        cloned_model.set_weights(self.model.get_weights())
        return cloned_model
    

    def __preprocess_image(self, img, std=0.1):
        img = np.copy(img)

        img -= img.mean()
        img /= (img.std() + 1e-5)
        img *= std

        # clip to [0, 1]
        img += 0.5
        img = np.clip(img, 0, 1)

        # convert to RGB array
        img *= 255
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def __imshow(self, img, preprocess=True, *args, **kwargs):
        
        img = img[0] if len(img.shape) == 4 else img  

        if preprocess:
            img = self.__preprocess_image(img, std=0.1)
        plt.imshow(img)

    def visualize(self, data, method, *args, **kwargs):
        if method == "vanilla":
            res = self.vanilla(data)
        elif method == "deconvnet":
            res = self.deconvnet(data)
        elif method == "guided":
            res = self.guided_backprop(data)

        self.__imshow(res, preprocess=True)
        return res

    def vanilla(self, data):
        if len(data.shape) == 3:
            data = data[np.newaxis, ...]
        elif len(data.shape) == 4:
            if data.shape[0] != 1:
                raise Exception("Can process only one image")
            
        new_data = tf.Variable(data, tf.float32)
        out_layer = self.model.get_layer(name=self.layer_name)

        with tf.GradientTape() as tape:
            tape.watch(new_data)

            out = tf.keras.models.Model(self.model.input, out_layer.output)(new_data, training=False)[0]
            neuron = tf.math.reduce_max(out)

        gradients = tape.gradient(neuron, new_data).numpy()
        # gradients = tf.math.reduce_max(tf.math.abs(gradients), axis=-1, keepdims=True).numpy()

        return gradients
    
    def deconvnet(self, data):

        if len(data.shape) == 3:
            data = data[np.newaxis, ...]
        elif len(data.shape) == 4:
            if data.shape[0] != 1:
                raise Exception("Can process only one image")

        deconv_model = self.__clone_model()
        for layer in deconv_model.layers:
            if isinstance(layer, tf.keras.layers.Activation):
                if layer.activation.__name__ == "relu":
                    layer.activation = DeconvReLU

        new_data = tf.Variable(data, tf.float32)
        out_layer = deconv_model.get_layer(name=self.layer_name)

        with tf.GradientTape() as tape:
            tape.watch(new_data)

            out = tf.keras.models.Model(deconv_model.input, out_layer.output)(new_data, training=False)[0]
            neuron = tf.math.reduce_max(out)

        gradients = tape.gradient(neuron, new_data).numpy()

        return gradients
    

    def guided_backprop(self, data, index=None):

        if len(data.shape) == 3:
            data = data[np.newaxis, ...]
        elif len(data.shape) == 4:
            if data.shape[0] != 1:
                raise Exception("Can process only one image")

        guided_model = self.__clone_model()
        for layer in guided_model.layers:
            if isinstance(layer, tf.keras.layers.Activation):
                if layer.activation.__name__ == "relu":
                    layer.activation = GuidedReLU

        new_data = tf.Variable(data, tf.float32)
        out_layer = guided_model.get_layer(name=self.layer_name)

        with tf.GradientTape() as tape:
            tape.watch(new_data)

            out = tf.keras.models.Model(guided_model.input, out_layer.output)(new_data, training=False)[0]

            if index is None:
                neuron = tf.math.reduce_max(out)
            else:
                neuron = out[index]
        gradients = tape.gradient(neuron, new_data).numpy()

        return gradients

