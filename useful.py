import numpy as np
from camera import to2D
from numpy.linalg import inv

# Generates 9 global points at (-,+,+) sector
def generate9globalPoints():
    glob = np.random.randint(1, 7, size=(9, 3))
    glob[:, 0] = -glob[:, 0]
    return glob


# make absolute image points from global cords - is used for visualithation
def make_absolute_image_points(camera, cords):
    get_rotated_cords = lambda Rt, X, p: (inv(Rt) @ to2D(Rt @ (p - X))) + X
    Rt = camera.getBasis()
    X = camera.getOffset()
    result = [[]] * len(cords)
    for i in range(0, len(cords)):
        result[i] = get_rotated_cords(Rt, X, cords[i])
    return np.array(result)
