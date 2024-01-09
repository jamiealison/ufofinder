# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:29:55 2024

@author: au694732

Remember to:

pip install miniball
pip install opencv-python

As conda cannot install those packages.

"""

import numpy as np, miniball, math, cv2, scipy.ndimage, os

wd="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/ScreenDumps/2023 08 10-16/"

# List all files in the directory ending with ".jpg"
jpg_files = [f for f in os.listdir(wd) if f.lower().endswith(".jpg")]

files=jpg_files[0:100]

for file in files:

    # Load the image
    image = cv2.imread(wd+file)
    
    this_mask = np.all((image >= [100,0,0]) & (image <= [120,5,5]), axis=-1)
    if file==files[0]:
        mask = np.all((image >= [100,0,0]) & (image <= [120,5,5]), axis=-1)
    mask = mask | this_mask
    
    
# b=image[:, :, 0]
# r=image[:, :, 2]


# ret,scanb = cv2.threshold(b,100,120,cv2.THRESH_BINARY)
# ret,scanr = cv2.threshold(r,0,5,cv2.THRESH_BINARY)

#need to shrink and extract only the edge pixels to avoid memory error when finding minimum bounding circle
mask_shrink=scipy.ndimage.binary_erosion(mask)
mask_edge=mask!=mask_shrink
mask_edge_disp=(mask_edge * 255).astype(np.uint8)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", mask_edge_disp)
cv2.waitKey()

S = np.argwhere(mask_edge)
print(len(S))


np.random.seed(1)
#S = np.random.randn(100, 2)
C, r2 = miniball.get_bounding_ball(S)
print("Minimum Bounding Circle Center:", C)
print("Minimum Bounding Circle Radius squared:", math.sqrt(r2))

# Draw a circle on the image
color = (0, 255, 0)  # Green color (BGR format)
thickness = 2  # Thickness of the circle outline
#[::-1] is the reverse the order of the centre pixels...
cv2.circle(image, np.round(C[::-1]).astype(int), np.round(math.sqrt(r2)-0.5).astype(int), color, thickness)

# Display the image
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.waitKey()

output_path = wd+"an_output_image.jpg"  # Replace with the desired output path and file name

#convert boolean array to binary image to display
mask_disp=(mask * 255).astype(np.uint8)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", mask_disp)
cv2.waitKey()

# Save the image as JPEG
cv2.imwrite(output_path, mask_disp)
