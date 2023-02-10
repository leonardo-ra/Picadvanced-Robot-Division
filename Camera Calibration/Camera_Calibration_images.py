import math
from statistics import mean, median, mode
import numpy as np
import cv2
import glob
from numpy import linalg as LA

# Board Size
board_h = 9
board_w = 6

# Arrays to store object points and image points from all the images_left.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


def  FindAndDisplayChessboard(img):
    # Find the chess board corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)
    # Set the needed parameters to find the refined corners
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
    # Calculate the refined corner locations
    corners = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)

    if ret == True:
        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)

    return ret, corners

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_w*board_h,3), np.float32)
objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

# Arrays to store object points and image points from all the images_left.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Read images_left
images = glob.glob('./Small/frame*.jpg')
# images = glob.glob('./Big/frame*.jpg')

for fname in images:
    img = cv2.imread(fname)
    ret, corners = FindAndDisplayChessboard(img)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None, None)

print(ret)
print ( " Intrinsics : " )
print (mtx)
print ( " Distortion : " )
print ( dist )
for i in range ( len ( tvecs ) ) :
    print ( " Translations (% d ) : " % i )
    print ( tvecs [0])
    print ( " Rotation (% d ) : " % i )
    print ( rvecs [0])


np.savez('camera_image_small.npz', cameraMatrix = mtx , distortion = dist, rvecs = rvecs, tvecs = tvecs)
cv2.waitKey(-1)
cv2.destroyAllWindows()

