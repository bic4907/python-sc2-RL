import random
import time
import sc2
from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.player import Bot, Computer
from sc2.unit_command import UnitCommand
from sc2.position import Point2, Point3

from UnitFeature import UnitFeature
from Net import Net
from Macro import Macro

class RLAgent(sc2.BotAI):

    # Features
    f_unit = None
    net = None
    macro = None

    def on_start(self):
        self.macro = Macro(self)

        self.f_unit = UnitFeature(self)
        f_unit_shape = self.f_unit.get_feature_shape()
        f_unit_shape_flatten = f_unit_shape.reshape([-1, 1])

        self.available_actions = ['NO_OP', 'TRAIN_SCV', 'TRAIN_MARINE', 'BUILD_SUPPLYDEPOT', 'BUILD_BARRACK']

        self.net = Net(f_unit_shape_flatten.shape[0] , len(self.available_actions))
        self.net.global_step = 0

    async def on_step(self, iteration):
        if self.state.game_loop > 10 and self.state.game_loop >= 14300:
            await self._client.restart()

        f_unit = self.f_unit.render(show=True)

        if iteration % 8 != 0:
            return



        self.net.global_step = self.state.game_loop / 8

        f_unit_shape_flatten = f_unit.reshape([-1, 1])

        action = self.net.get_action(f_unit_shape_flatten)
        print('Macro run', self.available_actions[action], "at", iteration)
        await self.macro.run(action)









    def on_end(self, result):
        pass





def main():
    sc2.run_game(sc2.maps.get("BuildMarines"), [
        Bot(Race.Terran, RLAgent())
    ], realtime=False)


if __name__ == '__main__':
    main()
