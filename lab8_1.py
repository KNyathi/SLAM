import numpy as np
import cv2 as cv
import glob

# Define grid size (number of circles)
pattern_size = (6, 10)  
flags = cv.CALIB_CB_SYMMETRIC_GRID  # Change to cv.CALIB_CB_ASYMMETRIC_GRID if using an asymmetric pattern

# Termination criteria for subpixel refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for the circular grid
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points
imgpoints = []  # 2D points


# Load images
images = glob.glob('Circular/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the circle grid pattern
    ret, centers = cv.findCirclesGrid(gray, pattern_size, flags=flags)

    if ret:
        objpoints.append(objp)
        refined_centers = cv.cornerSubPix(gray, centers, (11, 11), (-1, -1), criteria)
        imgpoints.append(refined_centers)

        # Draw the detected circles
        cv.drawChessboardCorners(img, pattern_size, refined_centers, ret)
        cv.imshow('Detected Circles', img)
        cv.waitKey(5000)

print(f"Number of valid images used for calibration: {len(objpoints)}")
if len(objpoints) == 0:
    print("❌ No valid images detected. Check the pattern size, images, and lighting.")

# Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

                                   
# Save calibration results
np.savez('calib_circles.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Load an image for undistortion
img = cv.imread('Board/left12.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Undistort the image
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calib_circles_result.png', dst)

# Alternative undistortion using remap
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
dst = dst[y:y+h, x:x+w]
cv.imwrite('calib_circles_result2.png', dst)


# Calculate total error
mean_error = 0
for i in range(len(objpoints)):
    
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Total error: {mean_error / len(objpoints)}")

cv.destroyAllWindows()
