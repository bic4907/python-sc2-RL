import random
import time
import sc2
from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.player import Bot, Computer
from sc2.unit_command import UnitCommand
from sc2.position import Point2, Point3

from UnitFeature import UnitFeature
from Net import DQN
from Macro import Macro

class RLAgent(sc2.BotAI):

    # Features
    f_unit = None
    net = None
    macro = None

    # variables for RL
    obs = None


    def on_start(self):
        self.macro = Macro(self)

        self.f_unit = UnitFeature(self)
        f_unit_shape = self.f_unit.get_feature_shape()

        self.available_actions = ['NO_OP', 'TRAIN_SCV', 'TRAIN_MARINE', 'BUILD_SUPPLYDEPOT', 'BUILD_BARRACK']

        self.DQN = DQN(f_unit_shape.shape , len(self.available_actions))

        self.prev_obs = None
        self.prev_a = None
        self.prev_score = 0
    '''
        Variables
        s = prev_obs
        a = prev_a
        r = ??          # 지금은 sparse reward
        s' = obs
        done            # 시나리오 끝날 때 되면 done 수동처리 및 환경리셋
    '''
    async def on_step(self, iteration):
        self.DQN.step = self.state.game_loop / 8
        done = False
        r = self.state.score.score
        '''
        import pickle
        with open('C:\\Users\\Admin\\Desktop\\git\\python-sc2-RL\\RL\\playground\\observation2.pkl', 'wb') as f:
            pickle.dump(self.state.observation, f)
            f.close()
        exit()
        '''

        print(self.state.feature)


        #for key, value in self.state.feature['screen'].items():
        #    value.save_image(key + '.jpg')

        #print(self.state.feature['screen']['height_map'].save_image('HIHI.jpg'))
        # with open('C:\\Users\\Admin\\Desktop\\observation.txt', 'w') as f:
        #     f.write(str(self.state.observation))
        #     f.close()
        #     exit()

        if self.state.game_loop > 20 and self.state.game_loop >= 14000:
            done = True

        if iteration % 10 != 0:
            return

        f_unit = self.f_unit.render(show=False)
        obs = f_unit


        action = self.DQN.get_action(obs)

        if self.prev_obs is not None:
            self.DQN.push_to_buffer([self.prev_obs, self.prev_a, self.state.score.score - self.prev_score, obs, done])



        self.prev_obs = obs
        self.prev_a = action




        if done:
            await self._client.restart()
            self.DQN.clear_buffer()
            self.prev_score = 0
        else:
            #print('Macro run', self.available_actions[action - 1], "at", iteration)
            await self.macro.run(action)



    def on_end(self, result):
        pass




import math
def main():
    sc2.run_game(sc2.maps.get("BuildMarines"), [
        Bot(Race.Terran, RLAgent())
    ], realtime=False
    )


if __name__ == '__main__':
    main()
