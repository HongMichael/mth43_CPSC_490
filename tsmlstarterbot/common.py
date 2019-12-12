import math

# Max number of planets.
PLANET_MAX_NUM = 28
MAP_MAX_WIDTH = 384
MAP_MAX_HEIGHT = 256
SCALE_FACTOR = 8
NUM_IMAGE_LAYERS = 4 # layer 0 for planets, layer 1 for player ships, layer 2 for enemy ships, layer 3 for available docking spots

# Number of initial features per planet we have

def distance2(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def distance(x1, y1, x2, y2):
    return math.sqrt(distance2(x1, y1, x2, y2))
