import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
import pickle
import numpy as np
from keras.applications.vgg16 import VGG16

# Credit go to:
#   outrunner
#       https://www.kaggle.com/outrunner/use-keras-to-count-sea-lions/notebook

width=100
batch_size=100

# Load the data
trainX = np.load('input/Train.npy')
trainY = np.load('input/Train_label2.npy')
valX = np.load('input/Val.npy')
valY = np.load('input/Val_label2.npy')


# Define model
initial_model = VGG16(weights="imagenet", include_top=False, input_shape=(100,100,3))
last = initial_model.output
x = Flatten()(last)
x = Dense(1024)(x)
x = LeakyReLU(alpha=0.1)(x)
preds=Dense(5, activation='linear')(x)
model=Model(initial_model.input, preds)

print model.summary()

# Start training slowly:
optim = keras.optimizers.SGD(lr=1e-5, momentum=0.2)
model.compile(loss='mean_squared_error', optimizer=optim, metrics=['accuracy'])

# Add checkpoint to save weights after each 10 epochs
filepath1 = 'snapshot_pre_{epoch:01d}.h5'
checkpoint1 = ModelCheckpoint(filepath1, save_best_only=False, period=1)
callbacks_list1=[checkpoint1]
history1 = model.fit(trainX, trainY, batch_size=batch_size, epochs=8, 
                    callbacks=callbacks_list1, verbose=2,
                    validation_data=(valX, valY))

save_history1 = open('history_1_pre.log', 'wb')
pickle.dump(history1.history, save_history1)

# Then speed up:
optim = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=optim, metrics=['accuracy'])

filepath2 = 'snapshot_pre_2_{epoch:01d}.h5'
checkpoint2 = ModelCheckpoint(filepath2, period=1)
callbacks_list2=[checkpoint2]
history2 = model.fit(trainX, trainY, batch_size=batch_size, epochs=20, 
                    verbose=2, callbacks=callbacks_list2,
                    validation_data=(valX, valY))

save_history2 = open('history_2_pre.log', 'wb')
pickle.dump(history2.history, save_history2)

model.save_weights('model_weights_pre.h5', overwrite=True)

print(history2.history.keys())



