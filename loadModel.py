import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./", one_hot=True)
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import optimizers
from keras.optimizers import SGD
import keras.models 
from keras.models import load_model
from keras.models import model_from_json


if __name__ == "__main__":
	x_test,y_test= mnist.test.images,mnist.test.labels
	x_test = x_test.astype('float32')
	x_test = x_test.reshape(x_test.shape[0],28,28,1)
	my_model = model_from_json(open('./my_model_architecture.json').read()) 
	my_model.load_weights('./my_model_weights.h5')	 
	print(my_model.summary())
	num_x = x_test.shape[0]
	print(y_test[1:2])
	print(my_model.predict(x_test[1:2]))
	# for i in range(num_x):

		# print (y_test[i])
		# print ('***')
  #   	print(my_model.predict(x_test[i]))