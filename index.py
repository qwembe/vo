from vo2019.camera import Camera
from vo2019.mplot import mPlot

mcam = Camera(orient=[80, 80, 80])

print(mcam.getBasis())

mplt = mPlot()
mplt.show_camera_basis(mcam)
mplt.show()
