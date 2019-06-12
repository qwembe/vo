from camera import Camera
from mplot import mPlot

mcam = Camera(orient=[80, 80, 80])

print(mcam.getBasis())

print("ready to use")

mplt = mPlot()
mplt.show_camera_basis(mcam)
mplt.show()
