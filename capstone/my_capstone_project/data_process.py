from  sklearn.preprocessing import label_binarize
from subprocess import check_output
import skimage.feature
import numpy as np
import pandas as pd
import cv2
import os, sys
import matplotlib.pyplot as plt

# Credit go to: 
#   Radu Stoicescu on Kaggle discussion
#       https://www.kaggle.com/radustoicescu/count-the-sea-lions-in-the-first-image
#   and outrunner 
#       https://www.kaggle.com/outrunner/use-keras-to-count-sea-lions/notebook

# scale and patch
classes = ['0', '1', '2', '3', '4']
train_path = "Train"
file_names_train = os.listdir("Train/")
file_names_train  = sorted(file_names_train, key=lambda
        item: (int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))

val_path = 'Val'
file_names_val = os.listdir("Val/")
file_names_val = sorted(file_names_val, key=lambda
        item:(int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))

test_path='Test'
file_names_test = os.listdir('Test/')
file_names_test = sorted(file_names_test, key=lambda
        item:(int(item.partition('.')[0]) if item[0].isdigit() else float('inf'), item))

# Get dot coordinates and cut image to patches
def preprocess_data(path,filename):
    r = 0.4 # scale down
    width = 100     # patch size

    # read the Train and Train Dotted images
    image_1 = cv2.imread(path + "Dotted/" + filename)
    image_2 = cv2.imread(path + '/' + filename)
    img1 = cv2. GaussianBlur(image_1, (5,5), 0)

    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1, image_2)
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    # convert to grayscale to be accpeted by skimage.feature.blob_log
    image_5 = np.max(image_4, axis=2)

    # detect blobs
    blobs = skimage.feature.blob_log(image_5, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

    h, w, d = image_2.shape

    res = np.zeros((int((w*r)//width)+1, int((h*r)//width)+1, 5), dtype='int16')

    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        
        # get the color of the pixel from Train Dotted in the center of the blob
        b, g, R = img1[int(y)][int(x)][:]
        x1 = int((x*r)//width)
        y1 = int((y*r)//width)

        # decision tree to pick the class of the blob by looking at the color in Train Dotted

        if R > 225 and b < 25 and g < 25:       # RED
            res[x1, y1, 0] += 1
        elif R > 225 and b > 225 and g < 25:    # MAGENTA
            res[x1, y1, 1] += 1
        elif R < 75 and b < 50 and 150 < g < 200:       # GREEN
            res[x1, y1, 4] += 1
        elif R < 75 and 150 < b < 200 and g < 75:       # BLUE
            res[x1, y1, 3] += 1
        elif 60 < R < 120 and b < 50 and g < 75:        # BROWN
            res[x1, y1, 2] += 1

    ma = cv2.cvtColor((1 * (np.sum(image_1, axis=2) > 20)).astype('uint8'), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(image_2 * ma, (int(w*r), int(h*r)))

    h1, w1, d = img.shape
    
    trainX = []
    trainY = []

    for i in range(int(w1//width)):
        for j in range(int(h1//width)):
            trainY.append(res[i,j,:])
            trainX.append(img[j*width:j*width+width, i*width:i*width+width,:])

    return np.array(trainX), np.array(trainY)

def get_data(path, filename):
    trainX, trainY = preprocess_data(path,filename)
    print trainY
    np.random.seed(1004)
    randomize = np.arange(len(trainX))
    np.random.shuffle(randomize)
    trainX = trainX[randomize]
    trainY = trainY[randomize]
    print trainY
    return trainX, trainY

def create_data(path):
    # dataframe to store result
    file_names = os.listdir(path+'/')

    data = []
    label=[]

    for filename in file_names:
        print filename
        trainX, trainY = get_data(path,filename)
        for i in range(0,len(trainX)):
            data.append(trainX[i])
            label.append(trainY[i])

    np.save(path,data)
    np.save(path + '_label2', label)
    print 'Done'

create_data(train_path)


    
