import numpy as np
import tensorflow as tf
from tqdm import tqdm

@tf.custom_gradient
def DeconvReLU(x):
    def grad(dy):
        return tf.cast(dy >= 0, tf.float32) * dy
    return tf.nn.relu(x), grad

@tf.custom_gradient
def GuidedReLU(x):
    def grad(dy):
        return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0, tf.float32) * dy
    return tf.nn.relu(x), grad

def raw_features(model, layer_name, data, use_logits=False, batch_size=256):
    y_pred = model.predict(data, batch_size=batch_size, verbose=0).argmax(axis=-1)
    submodel = tf.keras.models.Model(model.input, model.get_layer(layer_name).output)
    features = submodel.predict(data, batch_size=batch_size, verbose=0)
    relevances = features
    return relevances, y_pred


def vanilla_gradient(model, layer_name, data, use_logits=False, batch_size=256):
    """ Computes the (vanilla) gradient of maximally activated neuron 
    (neuron with the highest confidence) w.r.t. features produced by layer `layer_name`.
    If `use_logits` is True, neuron with the highest logits is used instead. 
    This method assumes that the model architecture follows ...->Dense(n_classes)->Softmax structure.
    If `use_logits` is True, predictions from the penultimate layer are used as logits values. So 
    keep in mind to provide such model architecture in this scenario.

    Args:
        - model: model you want to apply this method on,
        - layer_name: gradient is computed w.r.t. features produced by this layer,
        - data: you need some data of course,
        - use_logits: whether to compute gradient of softmax or logits w.r.t. layer_name' features,
        - batch_size: batch_size

    Returns:
        relevances, y_pred
    """
    
    relevances = []
    y_pred =  []
    
    layer = model.get_layer(name=layer_name)
    model_outputs = model.output if use_logits is False else model.layers[-2].output
    submodel = tf.keras.models.Model(model.input, [layer.output, model_outputs])
        
    for start in range(0, len(data), batch_size):
        x = data[start : start + batch_size].copy()
        x = tf.cast(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            outputs = submodel(x, training=False)

            hidden_features, confidences = outputs
            predicted_classes = tf.math.argmax(confidences, axis=-1, output_type=tf.int32).numpy()
            confidences = tf.math.reduce_max(confidences, axis=-1)
            confidences *= confidences
        
        gradients = tape.gradient(confidences, hidden_features).numpy()
        
        relevances.extend(gradients)
        y_pred.extend(predicted_classes)
        
    relevances = np.array(relevances)
    y_pred = np.array(y_pred)
    
    return relevances, y_pred

def vanilla_gradient_FromGenerator(model, layer_name, data, use_logits=False, batch_size=256):
    """ Computes the (vanilla) gradient of maximally activated neuron 
    (neuron with the highest confidence) w.r.t. features produced by layer `layer_name`.
    If `use_logits` is True, neuron with the highest logits is used instead. 
    This method assumes that the model architecture follows ...->Dense(n_classes)->Softmax structure.
    If `use_logits` is True, predictions from the penultimate layer are used as logits values. So 
    keep in mind to provide such model architecture in this scenario.

    Args:
        - model: model you want to apply this method on,
        - layer_name: gradient is computed w.r.t. features produced by this layer,
        - data: you need some data of course,
        - use_logits: whether to compute gradient of softmax or logits w.r.t. layer_name' features,
        - batch_size: batch_size

    Returns:
        relevances, y_pred
    """
    
    relevances = []
    y_pred =  []
    
    layer = model.get_layer(name=layer_name)
    model_outputs = model.output if use_logits is False else model.layers[-2].output
    submodel = tf.keras.models.Model(model.input, [layer.output, model_outputs])
       
    # for start in range(0, len(data), batch_size):
        # x = data[start : start + batch_size].copy()

    for i in tqdm(range(len(data))):
        x, y = next(data)
        # x = tf.cast(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            outputs = submodel(x, training=False)

            hidden_features, confidences = outputs
            predicted_classes = tf.math.argmax(confidences, axis=-1, output_type=tf.int32).numpy()
            confidences = tf.math.reduce_max(confidences, axis=-1)
            confidences *= confidences
        
        gradients = tape.gradient(confidences, hidden_features).numpy()
        
        relevances.extend(gradients)
        y_pred.extend(predicted_classes)
        
    relevances = np.array(relevances)
    y_pred = np.array(y_pred)
    
    return relevances, y_pred


def deconvnet(model, layer_name, data, use_logits=False, batch_size=256):
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
        x = data[start : start + batch_size].copy()
        x = tf.cast(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            outputs = submodel(x, training=False)

            hidden_features, confidences = outputs
            predicted_classes = tf.math.argmax(confidences, axis=-1, output_type=tf.int32).numpy()
            confidences = tf.math.reduce_max(confidences, axis=-1)
        
        gradients = tape.gradient(confidences, hidden_features).numpy()
        
        relevances.extend(gradients)
        y_pred.extend(predicted_classes)
        
    relevances = np.array(relevances)
    y_pred = np.array(y_pred)
    
    return relevances, y_pred


def deconvnet_FromGenerator(model, layer_name, data, use_logits=False, batch_size=256):
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

    for i in tqdm(range(len(data))):
        x, y = next(data)
                
        with tf.GradientTape() as tape:
            outputs = submodel(x, training=False)

            hidden_features, confidences = outputs
            predicted_classes = tf.math.argmax(confidences, axis=-1, output_type=tf.int32).numpy()
            confidences = tf.math.reduce_max(confidences, axis=-1)
        
        gradients = tape.gradient(confidences, hidden_features).numpy()
        
        relevances.extend(gradients)
        y_pred.extend(predicted_classes)
        
    relevances = np.array(relevances)
    y_pred = np.array(y_pred)
    
    return relevances, y_pred



def guided_backpropagation(model, layer_name, data, use_logits=False, batch_size=256):
    relevances = []
    y_pred =  []
    
    guided_model = tf.keras.models.clone_model(model)
    guided_model.set_weights(model.get_weights())

    for layer in guided_model.layers:
        if isinstance(layer, tf.keras.layers.Activation):
            if layer.activation.__name__ == 'relu':
                layer.activation = GuidedReLU

    layer = guided_model.get_layer(name=layer_name)
    model_outputs = guided_model.output if use_logits is False else guided_model.layers[-2].output
    submodel = tf.keras.models.Model(guided_model.input, [layer.output, model_outputs])

    for start in range(0, len(data), batch_size):
        x = data[start : start + batch_size].copy()
        x = tf.cast(x, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            outputs = submodel(x, training=False)

            hidden_features, confidences = outputs
            predicted_classes = tf.math.argmax(confidences, axis=-1, output_type=tf.int32).numpy()
            confidences = tf.math.reduce_max(confidences, axis=-1)
        
        gradients = tape.gradient(confidences, hidden_features).numpy()
        
        relevances.extend(gradients)
        y_pred.extend(predicted_classes)
        
    relevances = np.array(relevances)
    y_pred = np.array(y_pred)
    
    return relevances, y_pred


def guided_backpropagation_FromGenerator(model, layer_name, data, use_logits=False, batch_size=256):
    relevances = []
    y_pred =  []
    
    guided_model = tf.keras.models.clone_model(model)
    guided_model.set_weights(model.get_weights())

    for layer in guided_model.layers:
        if isinstance(layer, tf.keras.layers.Activation):
            if layer.activation.__name__ == 'relu':
                layer.activation = GuidedReLU

    layer = guided_model.get_layer(name=layer_name)
    model_outputs = guided_model.output if use_logits is False else guided_model.layers[-2].output
    submodel = tf.keras.models.Model(guided_model.input, [layer.output, model_outputs])

    for i in tqdm(range(len(data))):
        x, y = next(data)
        
        with tf.GradientTape() as tape:
            outputs = submodel(x, training=False)

            hidden_features, confidences = outputs
            predicted_classes = tf.math.argmax(confidences, axis=-1, output_type=tf.int32).numpy()
            confidences = tf.math.reduce_max(confidences, axis=-1)
        
        gradients = tape.gradient(confidences, hidden_features).numpy()
        
        relevances.extend(gradients)
        y_pred.extend(predicted_classes)
        
    relevances = np.array(relevances)
    y_pred = np.array(y_pred)
    
    return relevances, y_pred



def LRP(model, layer_name, data, use_logits=False, batch_size=256):

    for starting_layer_idx,l in enumerate(model.layers):
        if l.name == model.get_layer(layer_name).name:
            break
    
    
    weights = []
    wanted_layers = []

    for layer in model.layers[starting_layer_idx:]:
        if isinstance(layer, tf.keras.layers.Dense):
            weights.append(layer.get_weights())
        if isinstance(layer, 
                      (tf.keras.layers.Activation,
                      tf.keras.layers.ReLU)):
            wanted_layers.append(layer)

    wanted_layers.append(model.layers[-2]) # append logits
    
    submodel = tf.keras.models.Model(model.input, [x.output for x in wanted_layers])
    activations = submodel.predict(data, verbose=0)

    confidences = model.predict(data)

    y_pred = tf.math.argmax(confidences, axis=-1, output_type=tf.int32).numpy()
    
    relevances = activations[-1].copy() # use logits
    relevances *= np.where(relevances >= relevances.max(axis=-1, keepdims=True), 1, 0)
    rel = []
    
    for i in range(1, len(activations))[::-1]:

        act_i = activations[i-1]
        act_j = activations[i]
        w = weights[i]

        z = act_j + 1e-12
        s = relevances / z
        c = s @ w[0].T
        relevances = act_i * c
        rel.append(relevances)
    
    # if return_all_relevances:
    #     return dict(relevance=rel[-1], all_relevances=rel), y_pred
    # else:
    #     return rel[-1], y_pred

    return rel[-1], y_pred