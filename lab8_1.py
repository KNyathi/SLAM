import numpy as np
import cv2 as cv
import glob

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for a circular grid
# Assuming a grid of 7x6 circles
grid_size = (10, 6) 
objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

jpg_images = glob.glob('Circular/' + '*.jpg')
png_images = glob.glob('Circular/' + '*.png')

# Combine the lists of jpg and png images
images = jpg_images + png_images

print("Images found:", images)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the circular grid corners
    ret, corners = cv.findCirclesGrid(gray, grid_size, flags=cv.CALIB_CB_ASYMMETRIC_GRID, blobDetector=cv.SimpleBlobDetector_create())
    # If found, add object points, image points (after refining them)
    print(f"Processing {fname}: Grid found = {ret}")
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, grid_size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

# Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Undistortion
img = cv.imread('Circular/im4.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibrecirc.png', dst)

# Undistort using remap
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibrecirc2.png', dst)

# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("Total error: {}".format(mean_error / len(objpoints)))

cv.destroyAllWindows()
