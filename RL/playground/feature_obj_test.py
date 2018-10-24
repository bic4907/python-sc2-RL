import pickle
import matplotlib.pyplot as plt
import cv2
import numpy
from PIL import Image
from sc2.pixel_map import PixelMap
import time
from enum import Enum

from sc2.pixel_map_feature import PixelMapFeature

with open('observation2.pkl', 'rb') as f:
    obs = pickle.load(f)
    f.close()
#print(obs.feature_layer_data.renders.unit_type)
print('START', time.time())
unit_type = obs.feature_layer_data.renders.unit_energy_ratio
pm = PixelMapFeature(unit_type, 'render', 'unit_energy_ratio')
print(pm.numpy.shape)


print('END', time.time())