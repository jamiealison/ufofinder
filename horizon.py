# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:27:34 2024

@author: au694732
"""

import numpy as np, miniball, math, cv2, scipy.ndimage, os

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

# Define a line (start and end points)
line = (990, 950, 990, 0)

wd="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/ScreenDumps/2023 08 10-16/"

# List all files in the directory ending with ".jpg"
jpg_files = [f for f in os.listdir(wd) if f.lower().endswith(".jpg")]

files=jpg_files[0:100]
file=files[0]

# Load the image
image = cv2.imread(wd+file)
mask = cv2.imread(wd+"an_output_image.png")

image[mask[:,:,0]==0]=0

# Display the image
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.waitKey()

# Save the image as JPEG
cv2.imwrite(wd+"an_output_image_masked.png", image)

# Get pixels that intersect the line
intersecting_pixels = pixels_on_line(image, line)

color = (0, 255, 0)  # Green color (BGR format)
thickness = 2  # Thickness of the circle outline
cv2.line(image, line[0:2], line[2:4], color, thickness) 

# Visualize the result
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.waitKey()
