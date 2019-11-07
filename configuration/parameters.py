'''
This script contains constant parameters, if you desire to tinker with the code
you can find all the value here, no need to go through all the project.
'''

import numpy as np

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 10

# matrix of camera parameters (made up but works quite well for me)
# set u0 and v0 as half of the screen resolution
u0 = 640.0 / 2
v0 = 480.0 / 2
# calibration matrix
CAMERA_CALIBRATION = np.array([[800, 0, u0], [0, 800, v0], [0, 0, 1]])
