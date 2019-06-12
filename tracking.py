from camera import Camera
import numpy as np
from numpy.linalg import norm,inv
import copy
from triangulation import triangulate_cords

# get ratio between (1-st and 2-nd) and (2-nd and 3-d) camera position
def get_scale_ratio(X1, X2):
    # Finding relative scale between two sets of triangulated points
    r = [];
    for i, z1 in enumerate(zip(X1, X2)):
        x1, y1 = z1
        for j, z2 in enumerate(zip(X1, X2)):
            x2, y2 = z2
            if i == j: continue
            if abs(norm(y1 - y2)) > 1e-6: r.append(norm(x1 - x2) / norm(y1 - y2))
    return np.median(r)


def get_solution(four_solutions, cord1, cord2):
    test_solution = ["-"] * 4
    thisA = []
    k = 0
    true_cam = Camera()
    for Rt, X in four_solutions:
        test_cam = Camera(offset=X * 4)
        # After huge amount time of testing - it's better to make inv rotation
        test_cam.set_NewBasis(inv(Rt))
        temp = triangulate_cords(Camera(), test_cam, cord1, cord2)
        if temp:
            test_solution[k] = "+"
            true_cam = copy.copy(test_cam)
            A = temp.copy()
        k = k + 1
    #     print(test_solution)
    counter = sum(c == "+" for c in test_solution)
    if counter != 1:
        raise IOError("Wrong solutions in get_solution func: Immposible to find real solution!", test_solution)
    else:
        return A, true_cam.getBasis(), true_cam.getOffset()
