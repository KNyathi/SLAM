import cv2
import numpy as np

# Create a blank white canvas
canvas_size = 500
img = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

# Define colors in BGR format
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)

# Define circle positions
center_blue = (250, 150)  # Top center
center_green = (160, 300)  # Bottom left
center_red = (330, 300)  # Bottom right
radius = 50
thickness = 40  # Circle thickness

# Draw the three main circles
cv2.circle(img, center_blue, radius, red, thickness)
cv2.circle(img, center_green, radius, green, thickness)
cv2.circle(img, center_red, radius, blue, thickness)


# Function to draw a linear cutout (polygon) for the circle
def draw_linear_cutout(center, angle, color):
    # Angle in degrees, calculate cut lines using the angle
    start_angle = angle
    end_angle = start_angle + 45  # Make it 1/8 of the circle (45 degrees)
    
    # Generate points for the cutout, we assume center and radius, and the angle
    pt1 = center
    pt2 = (int(center[0] + 100 * np.cos(np.radians(start_angle))),
           int(center[1] - 100 * np.sin(np.radians(start_angle))))
    pt3 = (int(center[0] + 100 * np.cos(np.radians(end_angle))),
           int(center[1] - 100 * np.sin(np.radians(end_angle))))
    
    points = np.array([pt1, pt2, pt3], np.int32)
    points = points.reshape((-1, 1, 2))
    
    # Fill the area to simulate a "cut"
    cv2.fillPoly(img, [points], color)

# Draw the linear cutouts facing the center
draw_linear_cutout(center_blue, 245, white)  # Top circle facing downward
draw_linear_cutout(center_green, 10, white)  # Bottom left circle facing right
draw_linear_cutout(center_red, 70, white)  # Bottom right circle facing left

#draw a polygon
#pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
#pts = pts.reshape((-1,1,2))
#cv2.polylines(image,[pts],True,(0,255,255))


# Choose a different font type, for example, cv2.FONT_HERSHEY_TRIPLEX
font = cv2.FONT_HERSHEY_TRIPLEX

# Put text on the image with increased thickness to make it bold
cv2.putText(img, 'OpenCV', (2, 480), font, 4, (0, 0, 0), 5, cv2.LINE_AA)

# Display the image
cv2.imshow('OpenCV Logo', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
