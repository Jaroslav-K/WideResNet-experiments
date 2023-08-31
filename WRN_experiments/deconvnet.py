import numpy as np
import tensorflow as tf
#from tqdm import tqdm

@tf.custom_gradient
def DeconvReLU(x):
    def grad(dy): 
        dy = tf.constant(dy, dtype=tf.float32)   
        return tf.cast(dy >= 0, tf.float32) * dy
    return tf.nn.relu(x), grad

def normalize_gradient(img):
        """
		Gradients computed tend to become pretty small, especially after many layers.
		So after each layer, multiply them with a constant to keep them in acceptable 
		range (if applicable).
		"""
        gap = img.max() - img.min()
        if (abs(gap) > 1.):
            return img
        amplitude = 1./gap
        img *= amplitude
        
        return img

def filter_gradient(x):
    """
    The gradients to be visualize has non-negative value.
    """
    x_abs = np.abs(x)
    x_max = np.amax(x_abs, axis=-1)
    return x_max

def deconvnet(model, layer_name, data, use_logits=False, batch_size=256): #layer_name, 
    relevances = []
    y_pred =  []
    
    guided_model = tf.keras.models.clone_model(model)
    guided_model.set_weights(model.get_weights())

    for layer in guided_model.layers:
        if isinstance(layer, tf.keras.layers.Activation):
            if layer.activation.__name__ == 'relu':
                layer.activation = DeconvReLU

    layer = guided_model.get_layer(name=layer_name)
    model_outputs = guided_model.output if use_logits is False else guided_model.layers[-2].output
    submodel = tf.keras.models.Model(guided_model.input, [layer.output, model_outputs])

    for start in range(0, len(data), batch_size):
        x = data[start : start + batch_size]#.copy()
        x = tf.cast(x, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            outputs = submodel(x, training=False) #submodel

            input_data, confidences = outputs
            predicted_classes = tf.math.argmax(confidences, axis=-1, output_type=tf.int32).numpy()
            confidences = tf.math.reduce_max(confidences, axis=-1)
        
        gradients = tape.gradient(confidences, input_data).numpy()

        if (gradients.min() != gradients.max()):
            gradients = [normalize_gradient(grad) for grad in gradients]
        
        norm_gradients = filter_gradient(gradients)

        relevances.extend(norm_gradients)
        y_pred.extend(predicted_classes)
        
    relevances = np.array(relevances)
    y_pred = np.array(y_pred)
    
    return relevances, y_pred