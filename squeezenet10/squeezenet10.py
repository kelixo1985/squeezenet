import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# FIRE Module
def fire_module(x, squeeze_filters, expand_filters, name=None):
    squeeze = layers.Conv2D(squeeze_filters, (1, 1), activation='relu', padding='valid', name=name + '/squeeze')(x)
    
    expand_1x1 = layers.Conv2D(expand_filters, (1, 1), activation='relu', padding='valid', name=name + '/expand1x1')(squeeze)
    expand_3x3 = layers.Conv2D(expand_filters, (3, 3), activation='relu', padding='same', name=name + '/expand3x3')(squeeze)
    
    x = layers.concatenate([expand_1x1, expand_3x3], axis=-1, name=name + '/concat')
    return x

def SqueezeNet10(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    name="squeezenet10",
):
    if input_shape is None:
        input_shape = (224, 224, 3)

    if input_tensor is None:
        img_input = keras.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            img_input = keras.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = layers.Conv2D(96, (7, 7), strides=(2, 2), activation='relu', padding='valid', name=name + '/conv1')(img_input)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name=name + '/maxpool1')(x)

    x = fire_module(x, squeeze_filters=16, expand_filters=64, name=name + '/fire2')
    x = fire_module(x, squeeze_filters=16, expand_filters=64, name=name + '/fire3')
    x = fire_module(x, squeeze_filters=32, expand_filters=128, name=name + '/fire4')

    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name=name + '/maxpool4')(x)

    x = fire_module(x, squeeze_filters=32, expand_filters=128, name=name + '/fire5')
    x = fire_module(x, squeeze_filters=48, expand_filters=192, name=name + '/fire6')
    x = fire_module(x, squeeze_filters=48, expand_filters=192, name=name + '/fire7')
    x = fire_module(x, squeeze_filters=64, expand_filters=256, name=name + '/fire8')

    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name=name + '/maxpool8')(x)

    x = fire_module(x, squeeze_filters=64, expand_filters=256, name=name + '/fire9')

    if include_top:
        x = layers.Dropout(0.5, name=name + '/drop9')(x)
        x = layers.Conv2D(classes, (1, 1), activation='relu', padding='valid', name=name + '/conv10')(x)
        x = layers.GlobalAveragePooling2D(name=name + '/avgpool10')(x)
        x = layers.Activation(classifier_activation, name=name + '/activation')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name=name + '/avgpool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name=name + '/maxpool')(x)

    model = keras.Model(img_input, x, name=name)

    if weights == 'imagenet':
        raise ValueError("No pretrained weights available yet for SqueezeNet in Keras Applications.")
    
    return model