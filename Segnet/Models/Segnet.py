import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
import matplotlib.pyplot as plt
from keras import regularizers
import numpy as np




from keras.models import *
from keras.layers import *


import os
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = file_path+"/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"


def Segnet( n_classes ,  input_height=416, input_width=608 , level=3):

    image_input = Input(shape=(3,input_height,input_width))
    
    layer = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first' )(image_input)
    layer = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_first' )(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first' )(layer)
    level1 = layer
    	# Block 2
    layer = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_first' )(layer)
    layer = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_first' )(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_first' )(layer)
    level2 = layer
    
    	# Block 3
    layer = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_first' )(layer)
    layer = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_first' )(layer)
    layer = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_first' )(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_first' )(layer)
    level3 = layer
    
    	# Block 4
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_first' )(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_first' )(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_first' )(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_first' )(layer)
    level4 = layer
    
    	# Block 5
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_first' )(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_first' )(layer)
    layer = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_first' )(layer)
    layer = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_first' )(layer)
    level5 = layer
    
    layer = Flatten(name='flatten')(layer)
    layer = Dense(4096, activation='relu', name='fc1')(layer)
    layer = Dense(4096, activation='relu', name='fc2')(layer)
    layer = Dense( 1000 , activation='softmax', name='predictions')(layer)
    
    seg  = Model(  image_input , layer  )
    seg.load_weights(VGG_Weights_path)
    
    levels = [level1 , level2 , level3 , level4 , level5 ]
    
    DEC = levels[ level ]
    	
    DEC = ( ZeroPadding2D( (1,1) , data_format='channels_first' ))(DEC)
    DEC = ( Conv2D(512, (3, 3), padding='valid', data_format='channels_first'))(DEC)
    DEC = ( BatchNormalization())(DEC)
    
    DEC = ( UpSampling2D( (2,2), data_format='channels_first'))(DEC)
    DEC = ( ZeroPadding2D( (1,1), data_format='channels_first'))(DEC)
    DEC = ( Conv2D( 256, (3, 3), padding='valid', data_format='channels_first'))(DEC)
    DEC = ( BatchNormalization())(DEC)
    
    DEC = ( UpSampling2D((2,2)  , data_format='channels_first' ) )(DEC)
    DEC = ( ZeroPadding2D((1,1) , data_format='channels_first' ))(DEC)
    DEC = ( Conv2D( 128 , (3, 3), padding='valid' , data_format='channels_first' ))(DEC)
    DEC = ( BatchNormalization())(DEC)
    
    DEC = ( UpSampling2D((2,2)  , data_format='channels_first' ))(DEC)
    DEC = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(DEC)
    DEC = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format='channels_first' ))(DEC)
    DEC = ( BatchNormalization())(DEC)
    
    
    DEC =  Conv2D( n_classes , (3, 3) , padding='same', data_format='channels_first' )( DEC )
    DEC_shape = Model(image_input , DEC ).output_shape
    outputHeight = DEC_shape[2]
    outputWidth = DEC_shape[3]
    
    DEC = (Reshape((  -1  , outputHeight*outputWidth   )))(DEC)
    DEC = (Permute((2, 1)))(DEC)
    DEC = (Activation('softmax'))(DEC)
    model = Model( image_input , DEC )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    
    return model




if __name__ == '__main__':
	m = Segnet( 101 )
	m.summary()
    #from keras.utils import plot_model
	#plot_model( m , show_shapes=True , to_file='model.png')

