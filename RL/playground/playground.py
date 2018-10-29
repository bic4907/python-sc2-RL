import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import nn
from scipy.misc import imread
'''
dataset = None
with open('data.pickle', 'rb') as f:
    dataset = pickle.load(f)

#plt.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

convert_dataset = []
for data in dataset:
    b, g, r = cv2.split(data.numpy())
    convert_dataset.append([b, g, r])

tc = torch.Tensor(convert_dataset)
print(tc.shape)



conv = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(32, 64, kernel_size=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Conv2d(32, 16, kernel_size=2, padding=1),
    nn.ReLU()
)



output = conv.forward(tc)
print(output.shape)
'''

# Settings for feature layer
feature = dict()
#print()
feature.('screen', object())
#feature.screen.height_map: 11

print(feature)








