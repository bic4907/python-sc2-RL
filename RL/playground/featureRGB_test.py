import pickle
import matplotlib.pyplot as plt
import cv2
import numpy
from PIL import Image
from sc2.pixel_map import PixelMap
import time
from enum import Enum
import numpy as np

import sc2.colors as colors
import sc2.static_data as static_data

with open('observation2.pkl', 'rb') as f:
    obs = pickle.load(f)
    f.close()

print('START', time.time())
unit_type = obs.feature_layer_data.renders.unit_type


map = bytearray(unit_type.data)
a = np.frombuffer(map, dtype=np.int32)
a = a.reshape(200, 200)


def unit_type(scale=None):
  """Returns a palette that maps unit types to rgb colors."""
  # Can specify a scale to match the api or to accept unknown unit types.

  palette_size = scale or max(static_data.UNIT_TYPES) + 1


  palette = colors.shuffled_hue(palette_size)
  assert len(static_data.UNIT_TYPES) <= len(colors.distinct_colors)
  for i, v in enumerate(static_data.UNIT_TYPES):

    palette[v] = colors.distinct_colors[i]
  return palette

palette = unit_type()
print(palette.shape)

img = Image.fromarray(a, 'I')
img.show()

x_size = a.shape[1]
y_size = a.shape[0]

rgb = np.zeros([y_size, x_size, 3])


for y in range(y_size):
    for x in range(x_size):
        color = palette[a[y][x]]
        rgb[y][x] = color




        #print(palette[a[i][j]])



print('END', time.time())
cv2.imshow('Color image', rgb)
cv2.waitKey(0)
img = Image.fromarray(rgb, mode='RGB')
img.show()



class FeatureRenderDataTypes():
    height_map              =   numpy.uint8
    visibility_map          =   numpy.uint8
    creep                   =   numpy.bool
    power                   =   numpy.bool
    player_id               =   numpy.uint8
    unit_type               =   numpy.int32
    selected                =   numpy.bool
    unit_hit_points         =   numpy.int32
    unit_hit_points_ratio   =   numpy.uint8
    unit_energy             =   numpy.uint32
    unit_energy_ratio       =   numpy.uint8
    unit_shield             =   numpy.uint32
    unit_shield_ratio       =   numpy.uint8
    player_relative         =   numpy.uint8
    unit_density_aa         =   numpy.uint8
    unit_density            =   numpy.uint8
    effects                 =   numpy.uint8

class FeatureMinimapDataTypes():
    height_map              =   numpy.uint8
    visibility_map          =   numpy.uint8
    creep                   =   numpy.bool
    camera                  =   numpy.bool
    player_id               =   numpy.uint8
    player_relative         =   numpy.uint8
    selected                =   numpy.bool

class FeatureDataTypes():
    render  =   FeatureRenderDataTypes
    minimap =   FeatureMinimapDataTypes