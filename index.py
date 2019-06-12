from camera import Camera
from mplot import mPlot
from essmat import *
from triangulation import *
from useful import *

glob = generate9globalPoints()

mcam1 = Camera(offset=[-3, -3, 1], orient=[90, 0, 180])
mcam2 = Camera(offset=[0, -3, 1], orient=[90, 15, 180])
mcam3 = Camera(offset=[2, -4, 0], orient=[90, 45, 180])
mcam4 = Camera(offset=[2, 2, 4], orient=[90, 90, 180])
mcam5 = Camera(offset=[0, 0, 0], orient=[90, 45, 180])

image1 = mcam1.glob2image(glob)
image2 = mcam2.glob2image(glob)
image3 = mcam3.glob2image(glob)
image4 = mcam4.glob2image(glob)
image5 = mcam5.glob2image(glob)

print("ready to use")

mplt = mPlot()
mplt.show_camera_basis(mcam1)
mplt.show_camera_basis(mcam2)
mplt.show_camera_basis(mcam3)
mplt.show_camera_basis(mcam4)
mplt.show_camera_basis(mcam5)
mplt.addPoints(make_absolute_image_points(mcam1,glob))
mplt.addPoints(make_absolute_image_points(mcam2,glob))
mplt.addPoints(make_absolute_image_points(mcam3,glob))
mplt.addPoints(make_absolute_image_points(mcam4,glob))
mplt.addPoints(make_absolute_image_points(mcam5,glob))
mplt.addPoints(glob, param="r")
mplt.show()

