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

MAX_EPISODE = 100000

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

        self.g_episode = 0

        self._client.game_step = 32
    '''
        Variables
        s = prev_obs
        a = prev_a
        r = ??          # 지금은 sparse reward
        s' = obs
        done            # 시나리오 끝날 때 되면 done 수동처리 및 환경리셋
    '''
    async def on_step(self, iteration):
        try:
            self.DQN.step = self.state.game_loop / 8

            done = False
            r = self.state.score.score - self.prev_score
            #print(self.state.player_result, episode_complete)
            #if episode_complete:
            #    await self._client.restart()
            #    self.prev_score = 0
            #    return


            if self.state.player_result:
                self.g_episode += 1
                self.prev_obs = None
                self.prev_a = None
                self.prev_score = 0
                print('episode : %6d\t|\taccum_reward : %5d\t' % (self.g_episode, self.state.score.score))
                await self._client.restart()


            obs = self.state.feature['render']['unit_type'].numpy
            action = self.DQN.get_action(obs)

            if self.prev_obs is not None:
                self.DQN.push_to_buffer(self.prev_obs, self.prev_a, r, obs, done)

            if self.DQN.is_replay_full() and self.state.game_loop % 200 == 0:
                pass
                self.DQN.train()

            if self.DQN.is_replay_full() and self.state.game_loop % 400 == 0:
                self.DQN.update_target()


            self.prev_obs = obs
            self.prev_a = action
            self.prev_score = self.state.score.score

            q_value = self.DQN.get_probs(obs)
            debug = 'NO_OP            \t\t' + str(q_value[0]) + '\nTRAIN_SCV        \t\t' + str(q_value[1]) + '\nTRAIN_MARINE     \t\t' + str(q_value[2]) + '\nBUILD_SUPPLYDEPOT\t\t' + str(q_value[3]) + '\nBUILD_BARRACK    \t\t' + str(q_value[4]) + '\nREWARD    \t\t' + str(r) + '\nMEMORY    \t\t' + str(len(self.DQN.replay_buffer))
            self._client.debug_text_2d(text=debug, pos=Point2((0.05, 0.3)), size=14)
            await self._client.send_debug()


            await self.macro.run(action)
        except  KeyboardInterrupt:
            pass



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
