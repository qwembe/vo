import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class mPlot:

    def __init__(self):
        self.fig = plt.figure(figsize=[10, 8])
        self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        return

    @staticmethod
    def show_camera_basis(camera):
        camera, offset = camera.getBasis().copy(), camera.getOffset()
        zero = np.zeros(3)
        camera += offset
        x, y, z = zip(zero, offset)
        plt.plot(x, y, z, '--r')
        zero = offset
        x, y, z = zip(zero, camera[0])
        plt.plot(x, y, z, '-r')  # axis x
        x, y, z = zip(zero, camera[1])
        plt.plot(x, y, z, '-b')  # axis y
        x, y, z = zip(zero, camera[2])
        plt.plot(x, y, z, '-g')  # axis z

    def addPoints(self, dots, param='b', project=False):
        t = np.array(dots).T
        xs, ys, zs = t[0], t[1], t[2]
        self.ax.scatter(xs, ys, zs, c=param)
        if project:
            for cord in t.T:
                self.addVector([0, cord[1], cord[2]], cord, color="r", vectype="dashed", linewidths=0.4)
                self.addVector([cord[0], 0, cord[2]], cord, color="b", vectype="dashed", linewidths=0.4)
                self.addVector([cord[0], cord[1], 0], cord, color="g", vectype="dashed", linewidths=0.4)
        return

    def addVector(self, vec, offset=[0, 0, 0], color="r", vectype="solid", linewidths=0.9, vlenght=None):
        ax = self.ax
        mvec = np.array(vec) - np.array(offset)
        if vlenght is None:
            vlenght = np.linalg.norm(mvec)
        mvec = mvec / np.linalg.norm(mvec)
        ax.quiver(offset[0], offset[1], offset[2], mvec[0], mvec[1], mvec[2],
                  pivot='tail', length=vlenght, arrow_length_ratio=0.1,
                  colors=color, linestyles=vectype, linewidths=linewidths)

    def show(self, xs=-5, xf=5
             , ys=-5, yf=5
             , zs=-5, zf=5,
             showaxis=True):
        if showaxis:
            self.addVector([xf, 0, 0], offset=[xs, 0, 0], color="r", vectype="dashed", linewidths=0.5)
            self.addVector([0, yf, 0], offset=[0, ys, 0], color="b", vectype="dashed", linewidths=0.5)
            self.addVector([0, 0, zf], offset=[0, 0, zs], color="g", vectype="dashed", linewidths=0.5)
        self.ax.set_xlim(xs, xf)
        self.ax.set_ylim(ys, yf)
        self.ax.set_zlim(zs, zf)
        plt.show()
        return
