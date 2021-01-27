# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.layers import MaxPooling2D
from keras import models
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse
import utils
import os
import numpy as np
#from colorama import Fore, Back, Style
from keras.applications.vgg16 import VGG16, preprocess_input

# parsing arguments
def parse_args():
    parser = argparse.ArgumentParser(description='image classifier')
    parser.add_argument('--data', dest='data_dir', default='data')
    parser.add_argument('--list', dest='list_dir', default='list')
    parser.add_argument('--model', dest='model_name', required=True)
    args = parser.parse_args()
    return args

args = parse_args()

if utils.exist_list(args.list_dir):
    print('Lists exist in ./{0}. Use the test list.'.format(args.list_dir))
    classes, _, test_list = utils.load_lists(args.list_dir)
else:
    print('Lists do not exist.')
    exit(0)

test_image, test_label = utils.load_images(classes, test_list)
# convert to numpy.array
x_test = np.asarray(test_image)
y_test = np.asarray(test_label)
img = x_test
#img = np.expand_dims(img, axis=0)
#img = preprocess_input(img)
print("IMAGE: %s" % str(img.shape))   # IMAGE: (1, 224, 224, 3)
#print(img)
print('test samples: ', len(x_test))

model = load_model(args.model_name)
layers = model.layers[0:56]
layer_outputs = [layer.output for layer in layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
#activation_model.summary()

#'''
pred = model.predict(x_test, batch_size=32, verbose=0)
correct=0
incorrect=0
for i, test in enumerate(test_list):
    answer = os.path.basename(os.path.dirname(test))
    predict = classes[np.argmax(pred[i])]
    if answer == predict:
        print(test)
        print('Correct!!!' )
        correct=correct+1
        for j in range(len(classes)):
            if j == 3:
                print('')
            print('{0}: {1:4.2f} '.format(classes[j], pred[i][j]), end='')
        print('\n')
    else:
        print(test)
        print('Incorrect...' )
        incorrect=incorrect+1
        print('answer:', os.path.basename(os.path.dirname(test)))
        print('predict:', classes[np.argmax(pred[i])])
        for j in range(len(classes)):
            if j == 3:
                print('')
            print('{0}: {1:4.2f} '.format(classes[j], pred[i][j]), end='')
        print('\n')

print("total:",len(x_test),"incorrect:",incorrect)
print("test accuracy:",correct/(correct+incorrect)*100,"%")

activations = activation_model.predict(img)
for i, activation in enumerate(activations):
  print("%2d: %s" % (i, str(activation.shape)))

print("ここまではOK")

#print(max(activations[-2][0].transpose(2, 0, 1), key=lambda x: np.mean(x)).tolist())
'''
activations = [activation for layer, activation in zip(layers, activations) if isinstance(layer, MaxPooling2D)]

for i, activation in enumerate(activations):
  num_of_image = activation.shape[3]
  max = np.max(activation[0])
  for j in range(0, num_of_image):
    plt.figure()
    sns.heatmap(activation[0, :, :, j], vmin=0, vmax=max, xticklabels=False, yticklabels=False, square=False)
    plt.savefig("%d_%d.png" % (i+1, j+1))
    plt.close()
'''
# 出力層ごとに特徴画像を並べてヒートマップ画像として出力
activations = [(layer.name, activation) for layer, activation in zip(layers, activations) if isinstance(layer, MaxPooling2D)]

for i, (name, activation) in enumerate(activations):
  num_of_image = activation.shape[3]
  cols = math.ceil(math.sqrt(num_of_image))
  rows = math.floor(num_of_image / cols)
  screen = []
  for y in range(0, rows):
    row = []
    for x in range(0, cols):
        j = y * cols + x
        if j < num_of_image:
            row.append(activation[0, :, :, j])
        else:
            row.append(np.zeros())
    screen.append(np.concatenate(row, axis=1))
  screen = np.concatenate(screen, axis=0)
  plt.figure()
  sns.heatmap(screen, xticklabels=False, yticklabels=False)
  plt.savefig('maxpooling\{}.jpg'.format(name))
  #plt.savefig("%s.png" % name )

  plt.close()

#model.summary()