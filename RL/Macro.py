from sc2.constants import *


class Macro():
    def __init__(self, bot):
        self.env = bot

    async def run(self, action):
        if action == 0:
            self.NO_OP()
        elif action == 1:
            await self.TRAIN_SCV()
        elif action == 2:
            await self.TRAIN_MARINE()
        elif action == 3:
            await self.BUILD_SUPPLYDEPOT()
        elif action == 4:
            await self.BUILD_BARRACK()

    def NO_OP(self):
        pass

    async def TRAIN_SCV(self):
        cc = self.env.units(COMMANDCENTER).random
        await self.env.do_actions(actions=[cc.train(SCV)])

    async def BUILD_SUPPLYDEPOT(self):
        scv = self.env.units(SCV)
        if not scv:
            return
        scv = scv.random
        await self.env.build(SUPPLYDEPOT, near=self.env.units(COMMANDCENTER).random)


    async def BUILD_BARRACK(self):
        scv = self.env.units(SCV)
        if not scv:
            return
        scv = scv.random
        await self.env.build(BARRACKS, near=self.env.units(COMMANDCENTER).random)

    async def TRAIN_MARINE(self):
        br = self.env.units(BARRACKS)
        if not br:
            return
        br = br.random
        await self.env.do_actions(actions=[br.train(MARINE)])
