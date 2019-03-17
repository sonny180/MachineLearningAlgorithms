import tensorflow as tf
from tensorflow import keras
from keras import Model,Sequential
from keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D
from keras.layers import Input, MaxPooling2D, GlobalMaxPooling2D


def VGG16(num_classes,importModel = None):
	
	image_input = Input(shape = (50,50,3))
	#block1
	x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(image_input)
	x = Conv2D(64,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv2')(x)
	x = MaxPooling2D((2,2), strides = (2,2), name = 'block1_pool')(x)
	#block2
	x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)
	x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)
	x = MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')(x)
	#block3
	x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)
	x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)
	x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)
	x = MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')(x)
	#block4
	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv1')(x)
	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv2')(x)
	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv3')(x)
	x = MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')(x)
	#block5
	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv1')(x)
	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv2')(x)
	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv3')(x)	
	x = MaxPooling2D((2,2),strides = (2,2),name = 'block5_pool')(x)
	#Classification block
	x = Flatten(name = 'flatten')(x)
	x = Dense(4096,activation = 'relu',name = 'fc1')(x)
	x = Dense(4096,activation = 'relu',name = 'fc2')(x)
	x = Dense(num_classes,activation = 'softmax',name = 'fc3')(x)
	model = Model(image_input,x,name = 'vgg16')
	if importModel:
		model = Sequential()
		model.load_weights(importModel)
	return model

if __name__ == "__main__":
	newModel = VGG16(20)
	print(newModel.summary())
	importModel = VGG16(20,'vgg16_weights.h5')
	print(importModel.summary())

