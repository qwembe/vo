from camera import Camera
from mplot import mPlot
from essmat import *
from tracking import *
from useful import *
from triangulation import rotate_cords


# init stage make 3 cameras looking to global points
print()
glob = np.random.randint(1, 7, size=(9, 3))
glob[:, 0] = -glob[:, 0]

print("make cameras")
print("Camera 1")
camera1 = Camera([90, 0, 180], [-2, -2, 2])
print("Camera 2 ")
camera2 = Camera([90, 45, 180], [2, -2, 2])
print("Camera 3")
camera3 = Camera([90, 90, 180], [2, 1, 2])

image1 = camera1.glob2image(glob)
image2 = camera2.glob2image(glob)
image3 = camera3.glob2image(glob)

camera_cords1 = make_absolute_image_points(camera1, glob)
camera_cords2 = make_absolute_image_points(camera2, glob)
camera_cords3 = make_absolute_image_points(camera3, glob)
# end init

# Making essential matrix
print("Make essential matrix from camera1 and camera2")
E1 = createEssenceMatrix(image1, image2)
print("Make essential matrix from camera2 and camera3")
E2 = createEssenceMatrix(image2, image3)

# Calculating ratio:
four_solution_A = e_decomposition(E1)
four_solution_B = e_decomposition(E2)

A, Rt1, T1 = get_solution(four_solution_A, image1, image2)
B, Rt2, T2 = get_solution(four_solution_B, image2, image3)

mcam1 = Camera(offset=T1)
mcam2 = Camera(offset=T1 + inv(Rt1) @ T2)
mcam1.set_NewBasis(Rt1)
mcam2.set_NewBasis(Rt1 @ Rt2)
B = np.array(rotate_cords(mcam1, B))

#Calculate scale ratio
print(get_scale_ratio(A,B))


# make visual
true_image1 = rotate_cords(Camera(), image1)
true_image2 = rotate_cords(mcam1, image2)
true_image3 = rotate_cords(mcam2, image3)

mplt = mPlot()
mplt.show_camera_basis(Camera())
mplt.show_camera_basis(mcam1)
mplt.show_camera_basis(mcam2)

for i in range(0, len(image1)):
    mplt.addVector(A[i], Camera().getOffset(), vectype="dashed")  # ,vlenght=t)
    mplt.addVector(A[i], mcam1.getOffset(), vectype="dashed")  # ,vlenght=t)

for i in range(0, len(image2)):
    mplt.addVector(B[i], mcam1.getOffset(), vectype="dashed")  # ,vlenght=t)
    mplt.addVector(B[i], mcam2.getOffset(), vectype="dashed", color="b")  # ,vlenght=t)

mplt.addPoints(A, param='r')
mplt.addPoints(B, param='b')
# print(np.array(A)-np.array(B))

mplt.addPoints(true_image1, param='g')
mplt.addPoints(true_image2, param='b')
mplt.addPoints(true_image3, param='c')

mplt.show()
