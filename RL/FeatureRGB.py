
import sc2.colors as colors

class UnitFeature:

    palette = None
    obs = None
    width = None
    height = None

    def __init__(self, feature_map=None):
        self.palette = colors.unit_type()
        assert feature_map is not None
        self.raw = feature_map
        self.width = self.raw.shape[1]
        self.height = self.raw.shape[0]

    @property
    def numpy(self):
        pass

