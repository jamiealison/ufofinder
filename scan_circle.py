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

#folder="2023 08 10-16"
folder="Radar Grabs 2023 10 07 - 11"
indir="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/ScreenDumps/"+folder+"/"
outdir="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/Predictions/"+folder+"/"

# List all files in the directory ending with ".jpg"
jpg_files = [f for f in os.listdir(indir) if f.lower().endswith(".jpg")]
#print(jpg_files)
files=sorted(jpg_files)
files=[files[31]]

for file in files:
    
    print(file)
    # Load the image
    image = cv2.imread(indir+file)
    
    #eventually this probably needs to involve rolling through the images and using
    #an evolving background model?
    #it could be stored as a transparency layer on each image?
    this_blue = np.all((image >= [100,0,0]) & (image <= [120,1,1]), axis=-1)
    this_green = np.all((image >= [100,100,30]) & (image <= [150,150,70]), axis=-1)
    if file==files[0]:
        mask = np.copy(this_blue)
    mask = mask | this_blue
    mask = mask & np.logical_not(this_green)
    
    
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
xs = [t[0] for t in S]
ys = [t[1] for t in S]
print("Max x {}, min x {}.".format(max(xs),min(xs)))
print("Max y {}, min y {}.".format(max(ys),min(ys)))
warp=(max(xs)-min(xs))/(max(ys)-min(ys))
print(warp)

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

output_path = outdir+"a_scan_circle.png"  # Replace with the desired output path and file name

#convert boolean array to binary image to display
mask_disp=(mask * 255).astype(np.uint8)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", mask_disp)
cv2.waitKey()

# Save the image as png
cv2.imwrite(output_path, image)
