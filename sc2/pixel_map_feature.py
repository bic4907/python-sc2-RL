from typing import Callable, Set, FrozenSet, List
from .position import Point2

import numpy

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

class PixelMapFeature(object):
    def __init__(self, proto, type, attr):
        self._proto = proto
        self.data = bytearray(self._proto.data)
        _dtype = getattr(getattr(FeatureDataTypes, type), attr)
        data_np = numpy.frombuffer(self.data, dtype=_dtype)
        data_np = data_np.reshape([-1, self._proto.size.y, self._proto.size.x])
        self.convert = data_np

    @property
    def width(self):
        return self._proto.size.x

    @property
    def height(self):
        return self._proto.size.y

    @property
    def bits_per_pixel(self):
        return self._proto.bits_per_pixel

    @property
    def numpy(self):
        return self.convert

    def save_image(self, filename):
        assert NotImplementedError