# rotation = [0,0,90]
import numpy as np
from math import sin, cos
from numpy import degrees, arctan2

class Rotation:

    def __init__(self):
        return

    def Rx(self, degree):
        a = np.radians(degree)
        return np.array([[1, 0, 0],
                         [0, cos(a), -sin(a)],
                         [0, sin(a), cos(a)]])

    def Ry(self, degree):
        a = np.radians(degree)
        return np.array([[cos(a), 0, sin(a)],
                         [0, 1, 0],
                         [-sin(a), 0, cos(a)]])

    def Rz(self, degree):
        a = np.radians(degree)
        return np.array([[cos(a), -sin(a), 0],
                         [sin(a),  cos(a), 0],
                         [0, 0, 1]])

    def getCameraBasis(self, r):
        basis = np.eye(3)
        basis = self.Rx(r[0]) @ basis
        basis = self.Ry(r[1]) @ basis
        basis = self.Rz(r[2]) @ basis
        return basis

    #     @deprecated
    def getAnglesFromMatrix(R):
        alpha = degrees(arctan2(R[2][1], R[2][2]))
        beta = degrees(arctan2(-R[2][0], (R[2][1] ** 2 + R[2][2] ** 2) ** 0.5))
        gamma = degrees(arctan2(R[1][0], R[0][0]))
        return [alpha, beta, gamma]

    def rot_mat_to_euler(r):
        if (r[0, 2] == 1) | (r[0, 2] == -1):
            # special case
            e3 = 0  # set arbitrarily
            dlt = np.arctan2(r[0, 1], r[0, 2])
            if r[0, 2] == -1:
                e2 = np.pi / 2
                e1 = e3 + dlt
            else:
                e2 = -np.pi / 2
                e1 = -e3 + dlt
        else:
            e2 = -np.arcsin(r[0, 2])
            e1 = np.arctan2(r[1, 2] / np.cos(e2), r[2, 2] / np.cos(e2))
            e3 = np.arctan2(r[0, 1] / np.cos(e2), r[0, 0] / np.cos(e2))
        return [degrees(e1), degrees(e2), degrees(e3)]

