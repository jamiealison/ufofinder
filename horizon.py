# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:27:34 2024

@author: au694732
"""

import numpy as np, miniball, math, cv2, scipy.ndimage, os, statistics,skimage.measure,pandas

x_centre=990
y_centre=950
radius=929

def line_ends_from_angle(x_centre, y_centre, radius, angle):
    
    # Convert bearing from degrees to radians
    bearing_rad = math.radians(angle)

    # #change + to minus for both x and y because in images the positive y coordinate runs "south" so to speak
    
    # Calculate new latitude
    y_new = y_centre - radius * math.cos(bearing_rad)

    # Calculate new longitude
    x_new = x_centre + radius * math.sin(bearing_rad)

    return x_centre, y_centre, np.round(x_new).astype(int), np.round(y_new).astype(int)

#below is from chatgpt. I have tested it and it appears to work quite perfectly
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

files=sorted(jpg_files)
#file=files[1224]
file=files[0]
#file=files[1]
#there remains an issue with artefacts close to the centre. These may be best
#excluded by actually including the central blob in detection, and then eliminating
#based on size.
#file=files[6]
print(file)

# Load the image
image = cv2.imread(wd+file)
image_masked = np.copy(image)
image_angles = np.copy(image)
mask = cv2.imread(wd+"an_output_image.png")

image_masked[mask[:,:,0]==0]=0
 
rednesses=[]
#angles=[90]
angles=[x / 2.0 for x in range(0, 720)]
   
for angle in angles:

    # Define a line (start and end points)
    line=line_ends_from_angle(x_centre, y_centre, radius,angle)
    
    # Get pixels that intersect the line
    intersecting_pixels = pixels_on_line(image_masked, line)
    
    # Convert the list of coordinates to a NumPy array for indexing
    coordinates_array = np.array(intersecting_pixels)
    
    # Extract x and y coordinates separately
    x_coords, y_coords = coordinates_array[:, 0], coordinates_array[:, 1]
    
    # Modify pixel values using array indexing to highlight them
    # For example, setting the pixels to white (255, 255, 255)
    # using this to check for symmetry of the bresenham line function from chatgpt
    if(angle in [0]):
        image[y_coords, x_coords] = [255, 255, 255]

    redness=np.percentile(image_masked[y_coords, x_coords,2],40)
    
    # below is to plot the horizon lines
    if(redness>=1):
        #plot the line using cv2 functionality
        color = (0, 0, int(redness))  # Green color (BGR format)
        # for item in color:
        #     print(f"Item: {item}, Type: {type(item)}")
        thickness = 2  # Thickness of the circle outline
        cv2.line(image_angles, line[0:2], line[2:4], color, thickness) 
        
    rednesses=rednesses+[np.round(redness)]


horizon_r = angles[0:360][np.argmax(rednesses[0:360])]
horizon_l = angles[360:720][np.argmax(rednesses[360:720])]
print(horizon_r)
print(horizon_l)

print(rednesses)

horizon_r_clean=angles[np.nonzero(rednesses)[0][0] if np.any(rednesses) else None]-1
horizon_l_clean=angles[np.nonzero(rednesses)[0][-1] if np.any(rednesses) else None]+1
print(horizon_r_clean)
print(horizon_l_clean)
#here we use an artificial horizon to extract bad weather conditions

adj_horizon_r=horizon_r_clean
adj_horizon_l=horizon_l_clean
# adj_horizon_r=70
# adj_horizon_l=290

r_line=line_ends_from_angle(x_centre, y_centre, radius, adj_horizon_r)
r_pixels=pixels_on_line(image_masked, r_line)
l_line=line_ends_from_angle(x_centre, y_centre, radius, adj_horizon_l)
l_pixels=pixels_on_line(image_masked, l_line)
above_pixels=[]

for pixel in l_pixels+r_pixels:
    x, y = pixel
    y1=0
    while y1<=y:
        above_pixels.append((x, y1))
        y1+=1

# Convert the list of coordinates to a NumPy array for indexing
coordinates_array = np.array(above_pixels)

# Extract x and y coordinates separately
x_coords, y_coords = coordinates_array[:, 0], coordinates_array[:, 1]

horizon_mask=np.copy(mask)
horizon_mask[:]=0
horizon_mask[y_coords, x_coords] = [255]

mask=scipy.ndimage.binary_dilation(mask,iterations=1)
mask=mask&horizon_mask

image_masked[mask[:,:,0]==0,:]=0

# Save the image as JPEG
cv2.imwrite(wd+"an_output_image_masked.png", image_masked)
# Save the image as JPEG
cv2.imwrite(wd+"an_output_image_horizons.png", image_angles)

valid=np.transpose(np.where(mask[:,:,0] > 0))

#valid_r=np.mean(image_masked[:,:,2])
valid_r=np.percentile(image_masked[valid[:,0],valid[:,1],2], 95)
print(valid_r)

image_r=image_masked[:, :, 2]
# Save the image as JPEG
cv2.imwrite(wd+"an_output_image_r.png", image_r)

image_r_sm=scipy.ndimage.uniform_filter(image_r,size=3)

cv2.imwrite(wd+"an_output_image_r_sm.png", image_r_sm)

ret, image_r_th=cv2.threshold(image_r_sm,100,255,cv2.THRESH_BINARY)
cv2.imwrite(wd+"an_output_image_r_th.png", image_r_th)

image_ufos,n_ufos=skimage.measure.label(image_r_th,return_num=True)
cv2.imwrite(wd+"an_output_image_ufos.png", image_ufos+200)
print(n_ufos)

ufos_dict=skimage.measure.regionprops_table(image_ufos, properties=('label','centroid','bbox','area','axis_major_length','axis_minor_length'))
ufos=pandas.DataFrame(ufos_dict)
print(ufos)

ufos.to_csv(wd+'output_file.csv', index=False) 

ufos=ufos[ufos["area"]>3]

for ufo in range(0,len(ufos)):
    cv2.circle(image, (int(ufos["centroid-1"].iloc[ufo]),int(ufos["centroid-0"].iloc[ufo])), 10, (255,255,255), 3)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.waitKey()

cv2.imwrite(wd+"an_output_image_ufos_circled.png", image)
