import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

log1 = open('history_1_pre.log', 'rb')
log1 = pickle.load(log1)
log2 = open('history_2_pre.log', 'rb')
log2 = pickle.load(log2)

loss = log1['loss'] + log2['loss']
val_loss = log1['val_loss'] + log2['val_loss']
acc = log1['acc'] + log2['acc']
val_acc = log1['val_acc'] + log2['val_acc']

plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim([0,1])
plt.xlim([0,28])
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_figure.jpg')
plt.close()

