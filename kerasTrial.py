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


if __name__ == "__main__":
     print (mnist.train.images.shape,mnist.train.labels.shape)
     sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
     x_train,y_train = mnist.train.images,mnist.train.labels
     x_test,y_test= mnist.test.images,mnist.test.labels
     x_train = x_train.astype('float32')
     x_test = x_test.astype('float32')
     x_train = x_train.reshape(x_train.shape[0],28,28,1)
     x_test = x_test.reshape(x_test.shape[0],28,28,1)
     print (x_train.shape)
     print (y_train.shape)
     #build the model
     model = Sequential()
     model.add(Conv2D(32,(3,3),activation = 'relu',input_shape = (28,28,1),padding='same'))
     model.add(MaxPooling2D(pool_size = (2,2)))
     model.add(Conv2D(64,(3,3),activation = 'relu',padding='same'))
     model.add(MaxPooling2D(pool_size = (2,2)))
     model.add(Flatten())
     model.add(Dense(128,activation = 'relu'))
     model.add(Dense(10,activation = 'softmax'))
     print (model.summary())
     model.compile(loss=keras.losses.categorical_crossentropy,optimizer=sgd,metrics=['accuracy'])
     model.fit(x_train,y_train,batch_size = 100,epochs = 1)
     score = model.evaluate(x_test,y_test)
     print ("loss: "+str(score[0]))
     print ("accuracy: "+str(score[1]))
     #save architecture
     json_string = model.to_json()  
     open('my_model_architecture.json','w').write(json_string)
     #save weights
     model.save_weights('./my_model_weights.h5') 




 

