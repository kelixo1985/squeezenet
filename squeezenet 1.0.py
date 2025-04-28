import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, Activation, Dropout
from tensorflow.keras.models import Model

def fire_module(x, squeeze_filters, expand_filters):
    squeezed = Conv2D(squeeze_filters, (1, 1), activation='relu', padding='same')(x)
    expand1x1 = Conv2D(expand_filters, (1, 1), activation='relu', padding='same')(squeezed)
    expand3x3 = Conv2D(expand_filters, (3, 3), activation='relu', padding='same')(squeezed)
    return concatenate([expand1x1, expand3x3], axis=-1)

def SqueezeNet(input_shape, num_classes):
    input_img = Input(shape=input_shape)
    
    x = Conv2D(96, (7, 7), strides=(2, 2), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = fire_module(x, 32, 128)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = fire_module(x, 32, 128)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = fire_module(x, 64, 256)
    
    x = Dropout(0.5)(x)
    x = Conv2D(num_classes, (1, 1), padding='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax')(x)
    
    model = Model(inputs=input_img, outputs=output)
    return model

input_shape = (224, 224, 3) # Example input size, adjust as needed
num_classes = 7 # Number of skin lesion types in HAM10000
model = SqueezeNet(input_shape, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])