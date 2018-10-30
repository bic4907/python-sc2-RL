import pickle
import numpy as np
import cv2

import sc2.colors as colors

class UnitFeature():

    palette = None
    obs = None
    width = None
    height = None

    # feature_map is ImageData object
    def __init__(self, feature_map=None):
        self.palette = colors.unit_type()
        assert feature_map is not None
        self.width = feature_map.size.x
        self.height = feature_map.size.y
        self.raw = np.frombuffer(bytearray(feature_map.data), dtype=np.int32).reshape(self.height, self.width)

    @property
    def numpy(self):
        output = np.zeros([self.height, self.width, 3])
        for y in range(self.height):
            for x in range(self.width):
                color = self.palette[self.raw[y][x]]
                output[y][x] = color

        # Return shape : [height, width, 3(RGB)]
        return output

    # Change [height, width, 3(RGB)] to [3(RGB), height, width]
    @property
    def dataset(self):
        src = self.numpy
        return np.array(cv2.split(src))

    def show(self, w_name=''):
        img = self.numpy / 255
        cv2.imshow(w_name, img)
        cv2.waitKey(0)