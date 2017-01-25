# module:   dnn_image.py
# author:   Jin Yeom
# since:    01/24/17

import sys
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import SGD

''' Preprocessing image '''

filename = sys.argv[1]
img = Image.open(filename)

width, height = img.size
pixel_values = np.array(img)

X = np.array([[r, c] for r in range(width) for c in range(height)]) * 0.1
y = pixel_values.reshape(width*height, 1).astype('float32') / 255.

print 'Sample input shape: %s'  % str(X.shape)
print 'Sample output shape: %s' % str(y.shape)

''' Multilayer Perceptron (MLP) '''

num_hidden_layers = 8
num_hidden_neurons = 16

batch_size = 50
nb_epoch = 20

model = Sequential()
model.add(Dense(num_hidden_neurons, input_dim=2))
model.add(Activation('relu'))
model.add(Dropout(0.2))
for i in range(num_hidden_layers - 1):
    model.add(Dense(num_hidden_neurons))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epoch, 
          verbose=1, validation_data=(X, y))

score = model.evaluate(X, y, verbose=0)

print 'Test score: %f' % score[0]
print 'Test accuracy: %f' % score[1]

''' Reproducing image '''

outputs = model.predict(X, batch_size=batch_size).reshape(width, height)
repr_img = Image.fromarray((outputs * 255.).astype(np.uint8))
repr_img.save(filename+'_reproduced.png')
