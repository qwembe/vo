from camera import Camera
from essmat import createEssenceMatrix, e_decomposition
from mplot import mPlot
from tracking import get_solution, get_scale_ratio
from triangulation import rotate_cords
from numpy.linalg import inv
import numpy as np


#
# p* cords laying on the image
#

def process_vo(p0, p1, p2):
    image0 = np.array([[p[0]/1000, p[1]/1000, 1.0] for p in p0])
    image1 = np.array([[p[0]/1000, p[1]/1000, 1.0] for p in p1])
    image2 = np.array([[p[0]/1000, p[1]/1000, 1.0] for p in p2])
    # print(image0)
    # print(image1)
    # print(image2)
    # Making essential matrix
    # print("Make essential matrix from camera1 and camera2")
    E1 = createEssenceMatrix(image0, image1)
    # print("Make essential matrix from camera2 and camera3")
    E2 = createEssenceMatrix(image1, image2)

    four_solution_A = e_decomposition(E1)
    four_solution_B = e_decomposition(E2)

    A, Rt1, T1 = get_solution(four_solution_A, image0, image1)
    B, Rt2, T2 = get_solution(four_solution_B, image1, image2)

    mcam1 = Camera(offset=T1)
    mcam2 = Camera(offset=T1 + inv(Rt1) @ T2)
    mcam1.set_NewBasis(Rt1)
    mcam2.set_NewBasis(Rt1 @ Rt2)
    B = np.array(rotate_cords(mcam1, B))

    scale = get_scale_ratio(A,B)
    print(scale)

    # make visual
    true_image1 = rotate_cords(Camera(), image0)
    true_image2 = rotate_cords(mcam1, image1)
    true_image3 = rotate_cords(mcam2, image2)

    mplt = mPlot()
    mplt.show_camera_basis(Camera())
    mplt.show_camera_basis(mcam1)
    mplt.show_camera_basis(mcam2)

    for i in range(0, len(image0)):
        mplt.addVector(A[i], Camera().getOffset(), vectype="dashed")
        mplt.addVector(A[i], mcam1.getOffset(), vectype="dashed")

    for i in range(0, len(image1)):
        mplt.addVector(B[i], mcam1.getOffset(), vectype="dashed")
        mplt.addVector(B[i], mcam2.getOffset(), vectype="dashed", color="b")

    mplt.addPoints(A, param='r')
    mplt.addPoints(B, param='b')

    mplt.addPoints(true_image1, param='g')
    mplt.addPoints(true_image2, param='b')
    mplt.addPoints(true_image3, param='c')
    mplt.show(-50,50,-50,50,-50,50)
    return True, T1, scale
