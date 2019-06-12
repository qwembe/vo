import numpy as np
from numpy import cross
from numpy.linalg import inv


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


# move image cords from one cam to the absolute poition
def rotate_cords(camera, cords):
    get_rotated_cords = lambda Rt, X, p: (inv(Rt) @ p) + X
    Rt = camera.getBasis()
    X = camera.getOffset()
    result = [[]] * len(cords)
    for i in range(0, len(cords)):
        result[i] = get_rotated_cords(Rt, X, cords[i])
    return np.array(result)


def check_twist(base1, point1, base2, point2, X):
    ivec1 = point1 - base1
    ivec2 = point2 - base2
    xvec1 = X - base1
    xvec2 = X - base2
    if angle_between(ivec1, xvec1) < 0 or angle_between(ivec2, xvec2) < 0:
        raise IOError("twistied camera")
    else:
        return False


def find_positive_vector_intersection(base1, point1, base2, point2):
    vec1 = point1 - base1
    vec2 = point2 - base2
    n = cross(vec1, vec2)
    n1 = cross(vec1, n)
    n2 = cross(vec2, n)
    X1 = base1 + (((base2 - base1) @ n2) / (vec1 @ n2)) * vec1
    X2 = base2 + (((base1 - base2) @ n1) / (vec2 @ n1)) * vec2
    X = np.array((X1 + X2) / 2)
    check_twist(base1, point1, base2, point2, X)
    return X


# Triangulation from pcords1 on camera1 and pcords2 on camera2 position
def triangulate_cords(cam1, cam2, pcords1, pcords2):
    image1 = rotate_cords(cam1, pcords1)
    image2 = rotate_cords(cam2, pcords2)

    base_1 = cam1.getOffset()
    base_2 = cam2.getOffset()

    X1 = [[]] * len(image1)
    try:
        for i in range(0, len(image1)):
            X1[i] = find_positive_vector_intersection(base_1, image1[i], base_2, image2[i])
    except IOError as e:
        #         print("Exception ocurred: ",e)
        return False
    return X1