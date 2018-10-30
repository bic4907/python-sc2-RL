import pickle
import numpy as np
import cv2


import sc2.colors as colors

class UnitFeature:

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
        img = self.numpy



    def show(self):
        img = self.numpy / 255
        cv2.imshow('', img)
        cv2.waitKey(0)


if __name__ == '__main__':


    with open('playground/observation2.pkl', 'rb') as f:
        obs = pickle.load(f)
        f.close()

    unit_type = obs.feature_layer_data.renders.unit_type
    uf = UnitFeature(feature_map=unit_type)
    uf.show()
    uf.show()
