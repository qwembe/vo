from numpy.linalg import matrix_rank
from numpy import transpose

from camera import Camera
from mplot import mPlot
import numpy as np
from useful import make_absolute_image_points


def createEssenceMatrix(cordsCameraA, cordsCameraB):
    dot = cordsCameraA.shape[0]
    semiEss = [[]] * dot
    for i in range(dot):
        kron_product = np.kron(cordsCameraA[i], cordsCameraB[i]).T
        semiEss[i] = kron_product
    semiEss = np.array(semiEss)
    rank = matrix_rank(semiEss)
    if rank < 8:
        raise IOError("Two few mesurements", rank)
    # it works
    u,s,vt = np.linalg.svd(semiEss)
    e = vt[-1]
    e = np.reshape(e,(3,3))

    for i in range(0,dot):
        print(cordsCameraA[i] @ e @ cordsCameraB[i])

    return e


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
N = 1000
sigma = 2

glob = np.random.randint(1, 7, size=(N, 3))
glob[:, 0] = -glob[:, 0]

print("Camera 1")
camera1 = Camera([90, 0, 180], [-2, -2, 2])
print("Camera 2 ")
camera2 = Camera([90, 45, 180], [2, -2, 2])
print("Camera 3")
camera3 = Camera([90, 90, 180], [2, 2, 2])

camera1.setGlobalPoints(glob)
camera2.setGlobalPoints(glob)
camera3.setGlobalPoints(glob)

camera1.glob2image()
camera2.glob2image()
camera3.glob2image()

Acords = make_absolute_image_points(camera1, glob)
Bcords = make_absolute_image_points(camera2, glob)
Ccords = make_absolute_image_points(camera3, glob)

radnA = np.random.normal(0, sigma, (N, 3))
radnB = np.random.normal(0, sigma, (N, 3))
radnC = np.random.normal(0, sigma, (N, 3))

Acords = Acords + radnA
Bcords = Bcords + radnB
Ccords = Ccords + radnC

E = createEssenceMatrix(Acords, Bcords)

# print(det(E))
# print(radnA)
# print("Acords be like - \n",Acords)
# print("Bcords be like - \n",Bcords)
# print("Ccords be like - \n",Ccords)


# mplt = mPlot()
# mplt.addPoints(glob, param="r", project=True)
# mplt.addPoints(Acords, param="g")
# mplt.addPoints(Bcords, param="g")
# mplt.addPoints(Ccords, param="g")
# mplt.show_camera_basis(camera1)
# mplt.show_camera_basis(camera2)
# mplt.show_camera_basis(camera3)
#
# # mplt.addPoints(glob,param="r")
# mplt.show(xs=-7, xf=7,
#           ys=-7, yf=7,
#           zs=-7, zf=7)
