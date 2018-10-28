from sc2.constants import *
from sc2.position import Point2

class Macro():
    def __init__(self, bot):
        self.env = bot

    async def run(self, action, param=None):
        await self.default_action()

        if param[0] is not None:
            param[0] = int(param[0] * self.env.game_info.playable_area.width + self.env.game_info.playable_area.x)
            param[1] = int(param[1] * self.env.game_info.playable_area.height + self.env.game_info.playable_area.y)

        if action == 0:
            self.NO_OP()
        elif action == 1:
            await self.TRAIN_SCV()
        elif action == 2:
            await self.TRAIN_MARINE()
        elif action == 3:
            await self.BUILD_SUPPLYDEPOT(pos=param)
        elif action == 4:
            await self.BUILD_BARRACK(pos=param)

    async def default_action(self):
        # SCV
        try:
            await self.env.distribute_workers()
        except:
            pass

    def NO_OP(self):
        pass

    async def TRAIN_SCV(self):
        cc = self.env.units(COMMANDCENTER).random
        await self.env.do_actions(actions=[cc.train(SCV)])

    async def BUILD_SUPPLYDEPOT(self, pos=None):
        if pos is None: return
        scv = self.env.units(SCV)
        if not scv:
            return
        scv = scv.random
        await self.env.build(SUPPLYDEPOT, near=Point2((pos[0], pos[1])))


    async def BUILD_BARRACK(self, pos=None):
        if pos is None: return
        scv = self.env.units(SCV)
        if not scv:
            return
        scv = scv.random
        await self.env.build(BARRACKS, near=Point2((pos[0], pos[1])))

    async def TRAIN_MARINE(self):
        br = self.env.units(BARRACKS)
        if not br:
            return
        br = br.random
        await self.env.do_actions(actions=[br.train(MARINE)])
