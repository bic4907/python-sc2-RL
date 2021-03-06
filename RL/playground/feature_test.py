import pickle
import matplotlib.pyplot as plt
import cv2
import numpy
from PIL import Image
from sc2.pixel_map import PixelMap
import time
from enum import Enum
import numpy as np

import sc2.colors, sc2.static_data

with open('observation2.pkl', 'rb') as f:
    obs = pickle.load(f)
    f.close()
#print(obs.feature_layer_data.renders.unit_type)
print('START', time.time())
unit_type = obs.feature_layer_data.renders.unit_type


map = bytearray(unit_type.data)
a = np.frombuffer(map, dtype=np.int32)
print(a.shape)
a = a.reshape(200, 200)
print(a.shape)
print(a)

def unit_type(scale=None):
  """Returns a palette that maps unit types to rgb colors."""
  # Can specify a scale to match the api or to accept unknown unit types.
  palette_size = scale or max(static_data.UNIT_TYPES) + 1
  palette = shuffled_hue(palette_size)
  assert len(static_data.UNIT_TYPES) <= len(distinct_colors)
  for i, v in enumerate(static_data.UNIT_TYPES):
    print(i, v)
    palette[v] = distinct_colors[i]
  return palette

a = unit_type(a)




img = Image.fromarray(a, 'I')
img.show()

print('END', time.time())

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