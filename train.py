# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
sys.path.append('C:/Users/CHIHYUUUU/PycharmProjects/venv/Lib/site-packages')

from keras.models import Sequential                             # 用來啟用NN
from keras.layers import Dense, Dropout, Activation, Flatten    # Fully Connected Networks
from keras.layers import Convolution2D, MaxPooling2D            # Convolution Operation  # Pooling
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
#from keras import backend as K

import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import utils


SLASH = 0.1 # percentage of test(validation) data

# parsing arguments
def parse_args():
    parser = argparse.ArgumentParser(description='image classifier')
    parser.add_argument('--data', dest='data_dir', default='data')
    parser.add_argument('--list', dest='list_dir', default='list')
    args = parser.parse_args()
    return args
args = parse_args()

if utils.exist_list(args.list_dir):
    print('Lists already exist in ./{0}. Use these lists.'.format(args.list_dir))
    classes, train_list, test_list, val_list = utils.load_lists(args.list_dir)
else:
    print('Lists do not exist. Create list from ./{0}.'.format(args.data_dir))
    classes, train_list, test_list, val_list = utils.create_list(args.data_dir, args.list_dir, SLASH)

train_image, train_label = utils.load_images(classes, train_list)
test_image, test_label = utils.load_images(classes, test_list)
val_image, val_label = utils.load_images(classes, val_list)

# convert to numpy.array
x_train = np.asarray(train_image)
y_train = np.asarray(train_label)
x_test = np.asarray(test_image)
y_test = np.asarray(test_label)
x_val = np.asarray(val_image)
y_val = np.asarray(val_label)

print('train samples: ', len(x_train))
print('test samples: ', len(x_test))
print('validation samples: ', len(x_val))

NUM_CLASSES = len(classes)
BATCH_SIZE = 32
EPOCH = 50

# building the model
print('building the model ...')

model = Sequential()
model.add(Convolution2D(36, 3, 3, border_mode='same',input_shape=x_train.shape[1:]))#特徴数/ピクセル/ 卷積層1
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(36, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(36, 3, 3 , border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))                                 # 激活層
model.add(Convolution2D(36, 3, 3 , border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))                              # 池化層
model.add(Dropout(0.25))

model.add(Convolution2D(36, 3,3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(36, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(36, 3,3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(36, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(36, 3,3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(36, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())                                                    # 拉成一維數據
model.add(Dense(256))                                                   # 全連接層1
model.add(Activation('relu'))                                            #  激活層
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES))                                           # 全連接層2
model.add(Activation('softmax'))                                        #  激活層

rmsplop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=rmsplop, metrics=['accuracy'])

# training
hist = model.fit(x_train, y_train,
                 batch_size=BATCH_SIZE,
                 verbose=1,
                 nb_epoch=EPOCH,
                 validation_data=(x_val, y_val))

# save model
date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
model.save("TEST_Model"+date_str + '.model')

# plot loss
print(hist.history.keys())
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

nb_epoch = len(loss)
fig, ax1 = plt.subplots()
ax1.plot(range(nb_epoch), loss, label='loss', color='b')
ax1.plot(range(nb_epoch), val_loss, label='val_loss', color='g')
leg = plt.legend(loc='upper left', fontsize=10)
leg.get_frame().set_alpha(0.5)
ax2 = ax1.twinx()
ax2.plot(range(nb_epoch), acc, label='acc', color='r')
ax2.plot(range(nb_epoch), val_acc, label='val_acc', color='m')
leg = plt.legend(loc='upper right', fontsize=10)
leg.get_frame().set_alpha(0.5)
plt.grid()
plt.xlabel('epoch')
plt.savefig('graph_' + date_str + '.png')
plt.show()

model.summary()
#conv output
# 1st convolutional layer(modelの中での順番)
layer_num = 0
print('Layer Name: {}'.format(model.layers[layer_num].get_config()['name']))
W = model.layers[layer_num].get_weights()[0]

W = W.transpose(3, 2, 0, 1)
nb_filter, nb_channel, nb_row, nb_col = W.shape

# plot filter
plt.figure()
for i in range(nb_filter):
    im = W[i, 0]
    # scaling images
    scaler = MinMaxScaler(feature_range=(0, 255))
    im = scaler.fit_transform(im)

    plt.subplot(8, 8, i + 1)
    plt.axis('off')
    plt.imshow(im, cmap='gray')
plt.show()
