# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:27:34 2024

@author: au694732
"""

import numpy as np, miniball, math, cv2, scipy.ndimage, os, statistics

def line_ends_from_angle(x_centre, y_centre, radius, bearing_deg):
    
    # Convert bearing from degrees to radians
    bearing_rad = math.radians(bearing_deg)

    # Calculate new latitude
    #change + to minus for both x and y because in images the positive y coordinate runs "south" so to speak
    y_new = y_centre - radius * math.cos(bearing_rad)

    # Calculate new longitude
    x_new = x_centre - radius * math.sin(bearing_rad) / math.cos(math.radians(y_centre))

    return x_centre, y_centre, np.round(x_new).astype(int), np.round(y_new).astype(int)


def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return np.array(points)

def pixels_on_line(image, line):
    line_pixels = bresenham_line(*line)
    intersection_pixels = []

    for pixel in line_pixels:
        x, y = pixel
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            intersection_pixels.append((x, y))

    return intersection_pixels

wd="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/ScreenDumps/2023 08 10-16/"

# List all files in the directory ending with ".jpg"
jpg_files = [f for f in os.listdir(wd) if f.lower().endswith(".jpg")]

files=jpg_files[0:100]
file=files[0]

# Load the image
image = cv2.imread(wd+file)
image_masked = cv2.imread(wd+file)
mask = cv2.imread(wd+"an_output_image.png")

image_masked[mask[:,:,0]==0]=0
 
rednesses=[]
#angles=[90]
angles=[x / 2.0 for x in range(0, 721)]
   
for angle in angles:

    # Define a line (start and end points)
    line=line_ends_from_angle(990,950,929,angle)
    
    # Get pixels that intersect the line
    intersecting_pixels = pixels_on_line(image_masked, line)
    
    # Convert the list of coordinates to a NumPy array for indexing
    coordinates_array = np.array(intersecting_pixels)
    
    # Extract x and y coordinates separately
    x_coords, y_coords = coordinates_array[:, 0], coordinates_array[:, 1]
    
    # # Modify pixel values using array indexing to highlight them
    # # For example, setting the pixels to white (255, 255, 255)
    # image[y_coords, x_coords] = [255, 255, 255]
    
    redness=statistics.mean(image_masked[y_coords, x_coords,2])
    
    if(redness>1):
        #plot the line using cv2 functionality
        color = (0, 0, int(redness))  # Green color (BGR format)
        # for item in color:
        #     print(f"Item: {item}, Type: {type(item)}")
        thickness = 2  # Thickness of the circle outline
        cv2.line(image, line[0:2], line[2:4], color, thickness) 
        
    rednesses=rednesses+[redness]

# Visualize the result
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.waitKey()

# Save the image as JPEG
cv2.imwrite(wd+"an_output_image_masked.png", image)
