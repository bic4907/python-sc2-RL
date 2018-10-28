import random
import time
import sc2
from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.player import Bot,  Human
from sc2.unit_command import UnitCommand
from sc2.position import Point2, Point3

from UnitFeature import UnitFeature
from Net import DQN
from Macro import Macro

import torch

MAX_EPISODE = 100000

class RLAgent(sc2.BotAI):

    # Features
    f_unit = None
    net = None
    macro = None

    # variables for RL
    obs = None

    def __init__(self):
        self.macro = Macro(self)

        self.f_unit = UnitFeature(self)
        f_unit_shape = self.f_unit.get_feature_shape()

        self.available_actions = ['NO_OP', 'TRAIN_SCV', 'TRAIN_MARINE', 'BUILD_SUPPLYDEPOT', 'BUILD_BARRACK']
        self.require_position = [False, False, False, True, True]

        self.DQN = DQN(f_unit_shape.shape , len(self.available_actions))

        self.prev_obs = None
        self.prev_a = None
        self.prev_score = 0

        self.g_episode = 0

        # Variables for reward shaping
        self.prev_collected_minerals = 0
        self.prev_collected_minerals_efficiency = 0

    def on_start(self):
        self._client.game_step = 23

    '''
        Variables
        s = prev_obs
        a = prev_a
        r = ??          # 지금은 sparse reward
        s' = obs
        done            # 시나리오 끝날 때 되면 done 수동처리 및 환경리셋
    '''
    async def on_step(self, iteration):
        if (not self.state.player_result) and self.state.mineral_field.amount == 0:
            self._client.game_step = 200
            return

        self._client.game_step = 23



        if self.state.player_result:
            self.g_episode += 1
            self.prev_obs = None
            self.prev_a = None
            self.prev_score = 0
            self.prev_collected_minerals = 0
            self.prev_collected_minerals_efficiency = 0

            if self.state.mineral_field.amount:
                print('episode : %6d\t|\taccum_reward : %5d\t' % (self.g_episode, self.state.score.score))

            await self._client.reset()

            if self.g_episode % 50 == 0:
                try:
                    torch.save(self.DQN.m_net.state_dict(), 'save_model/' + str(self.g_episode) + '-m_net.pt')
                    torch.save(self.DQN.t_net.state_dict(), 'save_model/' + str(self.g_episode) + '-t_net.pt')
                except:
                    pass
            time.sleep(2)

        await self.macro.default_action()
        self.DQN.step = self.state.game_loop / self._client.game_step

        # Reward for Parameter
        collected = self.state.score.collected_minerals - self.prev_collected_minerals
        param_r = 0
        if collected < self.prev_collected_minerals_efficiency:
            param_r = -1

        # Reward for Action
        action_r = self.state.score.score - self.prev_score


        obs = self.state.feature['render']['unit_type'].numpy

        action, param = self.DQN.get_action(obs)

        #print(action, param)
        if self.prev_obs is not None:
            self.DQN.push_to_buffer(self.prev_obs, self.prev_a, action_r, param_r, obs)

        if self.DQN.step > 50000 == 0:
            self.DQN.train()

        if self.DQN.step % 10000 == 0:
            self.DQN.update_target()

        self.prev_obs = obs
        self.prev_a = action
        self.prev_param = param
        self.prev_score = self.state.score.score
        self.prev_collected_minerals = self.state.score.collected_minerals
        self.prev_collected_minerals_efficiency = collected


        q_value = self.DQN.get_probs(obs)
        debug = 'NO_OP            \t\t' + str(q_value[0]) + '\nTRAIN_SCV        \t\t' + str(q_value[1]) + '\nTRAIN_MARINE     \t\t' + str(q_value[2]) + '\nBUILD_SUPPLYDEPOT\t\t' + str(q_value[3]) + '\nBUILD_BARRACK    \t\t' + str(q_value[4]) + '\nACTION_REWARD    \t\t' + str(action_r) + '\nPRAM_REWARD    \t\t' + str(param_r) + '\nMEMORY    \t\t' + str(len(self.DQN.replay_buffer))
        self._client.debug_text_2d(text=debug, pos=Point2((0.05, 0.3)), size=14)
        await self._client.send_debug()


        try:
            await self.macro.run(action, param)
        except:
            pass



    def on_end(self, result):
        pass




def main():

    sc2.run_game(sc2.maps.get("BuildMarines"), [
        Bot(Race.Terran, RLAgent())
    ], realtime=False
    )
    '''
    sc2.run_game(sc2.maps.get("BuildMarines"), [
        Human(Race.Terran)
    ], realtime=True
    )
    '''

if __name__ == '__main__':
    main()
