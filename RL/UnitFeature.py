import numpy as np
import cv2

from sc2.constants import *

class UnitFeature():
    def __init__(self, botAI):
        self.env = botAI

        self.UNITCOLOR = dict()
        self.UNITCOLOR[MARINE] = (123, 255, 0)
        self.UNITCOLOR[SCV] = (255, 0 ,0)
        self.UNITCOLOR[COMMANDCENTER] = (255, 255 ,255)
        self.UNITCOLOR[SUPPLYDEPOT] = (165, 165, 0)
        self.UNITCOLOR[BARRACKS] = (165, 64, 165)


    def get_feature_shape(self):
        return np.zeros([35, 35, 3])

    def render(self, show=False):
        game_data = np.zeros((self.env.game_info.map_size[1], self.env.game_info.map_size[0], 3), dtype=np.uint8) # 3 for RGB

        #print(self.env.state.observation.map_state)

        for cc in self.env.units:


            pos = cc.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 0, self.UNITCOLOR[cc.type_id], -1) # -1 for filling the circle

        game_data = cv2.flip(game_data, 0)
        cropped = game_data[20:55, 15:50]
        # origin_resized = cv2.resize(game_data, dsize=None, fx=5, fy=5)
        # cropped_resized = cv2.resize(cropped, dsize=None, fx=5, fy=5)
        # cv2.imshow('UnitFeature_resized', origin_resized)


        if show:
            cv2.imshow('UnitFeature_cropped', cropped)
            cv2.waitKey(1)

        return cropped









