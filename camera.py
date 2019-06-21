"""
def transformCords(cords): - статическая функция
создает из локальных 3д координат - 2д
если точка находится за камерой (или за плоскостью, на которую проецируется, то функция возвращает None)


pCords1 = transformCords(cordsCamera1)
pCords2 = transformCords(cordsCamera2)
"""


import numpy as np
from numpy.linalg import inv
from rotation import Rotation


def to2D(cords):
    if cords[2] > 1:
        mc = cords.copy()
        test = np.array([mc[0] / cords[2], mc[1] / cords[2], 1])
        #         if not np.all(test >= -1.0) or not np.all(test <= 1.0):
        #             raise IOError("Точка находится вне объктива")
        return test
    else:
        raise IOError("Точка находится за изображением")


def transformCords(cords):
    p = [[]] * len(cords)
    try:

        for i in range(0, len(cords)):
            p[i] = to2D(cords[i])
        result = np.array(p)

        return result

    except IOError as e:
        p(e)
        return False


"""
class Camera - класс камера:
отвечает за ориетнацию камеры в пространстве
хранит базис и смещение

def bindingParams(self,camera) - вызывается, для того чтобы получить связующее смещение и поворот этой камеры относительно camera
возвращает матрицу доворота и необходимое смещение, для того чтобы совместить 1-ую со 2-ой камерой

def newCameraCords(self,gCords): - необходима для получения из глобальных координат новых в базисе этой камеры
на вход - глоб координаты
на выход - локальные координаты

"""


class Camera:

    def __init__(self,
                 orient=[0, 0, 0],
                 offset=[0, 0, 0]):
        self.offset = np.array(offset)
        self.R = Rotation()
        m_orient = np.array(orient)

        self.globalPoints = None
        self.imagePoints = False
        if m_orient.size == 3:
            self.basis = self.R.getCameraBasis(orient)
        else:
            self.basis = m_orient

    def set_offset(self, offset):
        self.offset = np.array(offset)

    def set_NewBasis(self, basis=np.eye(3)):
        self.basis = np.array(basis)

    def getOffset(self):
        return self.offset

    def getBasis(self):
        return self.basis

    # 1
    def setGlobalPoints(self, glob):
        self.globalPoints = glob

    # 2
    def getGlobalPoints(self):
        return self.globalPoints

    # 3
    def glob2image(self, glob=None):
        if glob is not None:
            self.setGlobalPoints(glob)
        if glob is None:
            assert ("Any points hasn't been found!")
        as_camera_see = self.newCameraCords(self.globalPoints)
        self.imagePoints = transformCords(as_camera_see)
        return self.imagePoints

    # 4
    def getImagePoints(self):
        return self.imagePoints

    # 5
    def setImagePoints(seld, imageCords):
        seld.imagePoints = imageCords

    def bindingParams(self, camera):
        basis = camera.getBasis()
        of = camera.getOffset()
        rR = inv(basis) @ self.basis
        dT = of - self.offset
        return rR.T, dT

    # makes global cords as it sees this cam
    def newCameraCords(self, gCords):
        gCords = np.array(gCords)
        cords = np.empty(gCords.shape)
        for i in range(gCords.shape[0]):
            temp = np.array([inv(self.basis.T) @ (gCords[i] - self.offset)])
            cords[i] = temp
        return cords


