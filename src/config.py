conf = {
    # Epsilon Greedy Strategy
    # Exploration rate is 100% when 1
    "EPS_START": 1.0,
    "EPS_END": 0.01,
    "EPS_DECAY": 0.001,
}


class Config:
    def __init__(self):
        self._config = conf

    def get(self, property_name):
        if property_name not in self._config.keys():
            return None
        return self._config[property_name]

