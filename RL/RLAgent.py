import random
import time
import sc2
from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.player import Bot, Computer
from sc2.unit_command import UnitCommand
from sc2.position import Point2, Point3

class RLAgent(sc2.BotAI):
    def on_start(self):
        pass

    async def on_step(self, iteration):
        if self.state.game_loop >= 1800:
            await self._client.restart()

        minerals = []
        for u in (self.state.observation.raw_data.units):
            if u.unit_type == NATURALMINERALS.value:
                minerals.append(Point2.from_proto(u.pos))

        actions = []
        for u in self.units(MARINE):
            pass
            actions.append(u.move(u.position.closest(minerals)))
        await self.do_actions(actions=actions)

        pass

    def on_end(self, result):
        pass





def main():
    sc2.run_game(sc2.maps.get("CollectMineralShards"), [
        Bot(Race.Terran, RLAgent())
    ], realtime=False)


if __name__ == '__main__':
    main()
