# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:27:34 2024

@author: au694732
"""

import numpy as np, miniball, math, cv2, scipy.ndimage, os, statistics,skimage.measure,pandas,time

x_centre=990
y_centre=950
radius=929
warp=0.989724175229854
radius2=int(radius*warp)
horizon_thresh=20
horizon_buff=2
min_s=60
target_h=30
interval=5

draw=False

folder="2023 08 10-16"
#folder="Radar Grabs 2023 10 07 - 11"
indir="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/ScreenDumps/"+folder+"/"
outdir="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/Predictions/"+folder+"/"

def line_ends_from_angle(x_centre, y_centre, radius, angle,warp):
    
    # Convert bearing from degrees to radians
    bearing_rad = math.radians(angle)

    # #change + to minus for both x and y because in images the positive y coordinate runs "south" so to speak
    
    # Calculate new latitude
    y_new = y_centre - (radius * math.cos(bearing_rad))*warp

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

def point_in_circle(x1,y1,x_centre,y_centre,radius,warp):
    dx = abs(x_centre - x1)
    dy = abs(y_centre - y1)
    d = math.sqrt(dx**2+(dy/warp)**2)
    return(d<radius)

def pixels_on_line(image, line):
    line_pixels = bresenham_line(*line)
    intersection_pixels = []

    for pixel in line_pixels:
        x, y = pixel
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            intersection_pixels.append((x, y))

    return intersection_pixels

def is_point_above_horizon(point, x_centre, y_centre, horizon_r, horizon_l):
    x, y = point
    x-=x_centre
    y=y_centre-y
    #note added reciprocal to math.tan to account for 0 being up
    if x<0:
        angle=horizon_l
        #taking the negative of the slope on the left side
        slope = -1/math.tan(math.radians(angle))  # Convert angle to radians and calculate slope
    else:
        angle=horizon_r
        slope = 1/math.tan(math.radians(angle))  # Convert angle to radians and calculate slope
        
    line_value = slope * abs(x)
    
    return y > line_value

# Example usage:
horizon_r = 82 # Replace with the angle of your line
horizon_l = 274.5
point_to_check = [1883, 842] # Replace with the coordinates of your point
print(1/math.tan(math.radians(horizon_r)))
print(-1/math.tan(math.radians(horizon_l)))
result = is_point_above_horizon(point_to_check, x_centre, y_centre, horizon_r, horizon_l)
print(result)

# Record start time
start_time = time.time()

if not os.path.exists(outdir):
    os.makedirs(outdir)

# List all files in the directory ending with ".jpg"
jpg_files = [f for f in os.listdir(indir) if f.lower().endswith(".jpg")]

files=sorted(jpg_files)

#file=files[1224]
#file=files[0]
#file=files[1]
#file=files[6]
#egFile=31
egFile=0
files=files[15:17]

print("1: initial setup:   "+(str(time.time()-start_time)))

for file in files[:]:

    print(file)
    
    # Load the image
    image = cv2.imread(indir+file)
    image_masked = np.copy(image)
    image_angles = np.copy(image)
    
    
    # Convert the image from BGR to HSV color space
    hls_image = cv2.cvtColor(image_masked, cv2.COLOR_BGR2HLS)
    #print(np.max(hls_image[:, :, 0]))
    
    h = hls_image[:, :, 0]
    s = hls_image[:, :, 2]
    l = hls_image[:, :, 1]
    l_x = 255-(2*np.abs(127.5-l.astype(float)))
    l_x = l_x.astype(np.uint8)
    s_cone=np.minimum(s,l_x)
    
    #cv2.imwrite(outdir+"an_output_image_hue.png", h)
    
    #cv2.imwrite(outdir+"an_output_image_s_cone.png", s_cone)
    
    s_below_min=s_cone<min_s
    h[s_below_min]=0
    
    # h_sm=scipy.ndimage.uniform_filter(h,size=3)
    # cv2.imwrite(wd+"an_output_image_r_sm.png", h_sm)
    
    h_th=np.logical_and(h>=target_h-interval,h<=target_h+interval).astype(int) * 255
    if file==files[egFile]:
        cv2.imwrite(outdir+"an_output_image_relevant.png", h_th)
     
    yellownesses=[]
    #angles=[90]
    angles=[x / 2.0 for x in range(0, 720)]
    
    print("2: image loaded and colorspace coverted   "+(str(time.time()-start_time)))
    
    for angle in angles:
    
        # Define a line (start and end points)
        line=line_ends_from_angle(x_centre, y_centre, radius, angle, warp)
        
        # Get pixels that intersect the line
        intersecting_pixels = pixels_on_line(h_th, line)
        
        # Convert the list of coordinates to a NumPy array for indexing
        coordinates_array = np.array(intersecting_pixels)
        
        # Extract x and y coordinates separately
        x_coords, y_coords = coordinates_array[:, 0], coordinates_array[:, 1]
        
        # Modify pixel values using array indexing to highlight them
        # For example, setting the pixels to white (255, 255, 255)
        # using this to check for symmetry of the bresenham line function from chatgpt
        # if(angle in [0]):
        #     image[y_coords, x_coords] = [255, 255, 255]
    
        #redness=np.percentile(image_masked[y_coords, x_coords,2],40)
        yellowness=np.mean(h_th[y_coords, x_coords])
        
        # below is to plot the horizon lines
        if(yellowness>horizon_thresh):
            #plot the line using cv2 functionality
            color = (0, 0, int(yellowness))  # Green color (BGR format)
            # for item in color:
            #     print(f"Item: {item}, Type: {type(item)}")
            thickness = 2  # Thickness of the circle outline
            cv2.line(image_angles, line[0:2], line[2:4], color, thickness) 
            
        yellownesses=yellownesses+[np.round(yellowness)]
    
    print("3: horizon lines detected   "+(str(time.time()-start_time)))
    
    horizon_r = angles[0:360][np.argmax(yellownesses[0:360])]
    horizon_l = angles[360:720][np.argmax(yellownesses[360:720])]
    print(horizon_r)
    print(horizon_l)
    
    yellownesses=np.array(yellownesses).astype(int)
    #print(yellownesses)
    
    horizon_r_clean=angles[np.where(yellownesses>horizon_thresh)[0][0] if np.any(yellownesses>horizon_thresh) else None]-horizon_buff
    horizon_l_clean=angles[np.where(yellownesses>horizon_thresh)[0][-1] if np.any(yellownesses>horizon_thresh) else None]+horizon_buff
    print(horizon_r_clean)
    print(horizon_l_clean)
    #here we use an artificial horizon to extract bad weather conditions
    
    adj_horizon_r=horizon_r_clean
    adj_horizon_l=horizon_l_clean
    #adj_horizon_r=70
    #adj_horizon_l=290
    
    r_line=line_ends_from_angle(x_centre, y_centre, radius, adj_horizon_r,warp)
    r_pixels=pixels_on_line(image_masked, r_line)
    l_line=line_ends_from_angle(x_centre, y_centre, radius, adj_horizon_l,warp)
    l_pixels=pixels_on_line(image_masked, l_line)
    above_pixels=[]
    
    for pixel in l_pixels+r_pixels:
        x, y = pixel
        y1=0
        while y1<=y:
            if point_in_circle(x,y1,x_centre,y_centre,radius,warp):
                above_pixels.append((x, y1))
            y1+=1
    
    # Convert the list of coordinates to a NumPy array for indexing
    coordinates_array = np.array(above_pixels)
    
    # Extract x and y coordinates separately
    x_coords, y_coords = coordinates_array[:, 0], coordinates_array[:, 1]
    
    horizon_mask=np.zeros_like(image)
    horizon_mask[y_coords, x_coords] = [255]
    
    print("4: horizon mask generated   "+(str(time.time()-start_time)))
    
    circle_mask = np.zeros_like(image)
    # cv2.circle(circle_mask, (x_centre, y_centre), radius, (255, 255, 255), thickness=-1)
    cv2.ellipse(circle_mask, (x_centre, y_centre), (radius,radius2), 0, 0, 360, (255, 255, 255), thickness=-1)
    
    image_masked=np.copy(image)
    image_masked[circle_mask[:,:,0]==0,:]=0
    
    horizon_mask=np.zeros_like(image)
    #-90 to the start angle as the function thinks 0 is east. +270 to the second angle to add 360 while accounting for the east problem. Second angle must be larger than the first as drawing is always clockwise.
    cv2.ellipse(horizon_mask, (x_centre, y_centre), (radius,radius2), 0, adj_horizon_l-90, adj_horizon_r+270, (255, 255, 255), thickness=-1)
    
    print("5: circle mask generated   "+(str(time.time()-start_time)))
    
    #important - we remove the areas outside the from the hue mask, but not the areas below the horizon
    h_th[circle_mask[:,:,0]==0,]=0
    
    if file==files[egFile]:
        cv2.imwrite(outdir+"an_output_image_segments.png", h_th)
        # Save the image as JPEG
        cv2.imwrite(outdir+"an_output_image_masked.png", image_masked)
        # Save the image as JPEG
        cv2.imwrite(outdir+"an_output_image_horizons.png", image_angles)
        cv2.imwrite(outdir+"an_output_image_horizon_mask.png", horizon_mask)
    
    valid=np.transpose(np.where(horizon_mask[:,:,0] > 0))
    
    #valid_r=np.mean(image_masked[:,:,2])
    valid_r=np.percentile(image_masked[valid[:,0],valid[:,1],2], 95)
    #print(valid_r)
    
    
    image_ufos,n_ufos=skimage.measure.label(h_th,return_num=True)
    print("6: UFOs detected before filtering: {},     {}".format(n_ufos,str(time.time()-start_time)))
    
    ufos_dict=skimage.measure.regionprops_table(image_ufos, properties=('label','centroid','bbox','area','axis_major_length','axis_minor_length'))
    ufos=pandas.DataFrame(ufos_dict)
    
    dx = abs(x_centre - ufos["centroid-1"])
    dy = abs(y_centre - ufos["centroid-0"])
    d = np.sqrt(dx**2+(dy/warp)**2)
    #note the distance is in strange units - "WARPED PIXELS"
    ufos=ufos.assign(distance=d)
    
    #filters
    #best do removal of close-by detections first
    out=d>50
    ufos=ufos[out]
    ufos["file"]=file
    ufos["horizon_l"]=horizon_l
    ufos["horizon_r"]=horizon_r
    pix_x=ufos["centroid-1"].astype(int).tolist()
    pix_y=ufos["centroid-0"].astype(int).tolist()
    ufos["above_horizon"]=horizon_mask[pix_y, pix_x,0] == 255
    # Example usage:
    angle_of_line = 30  # Replace with the angle of your line
    point_to_check = (3, 5)  # Replace with the coordinates of your point
    ufos['above_line'] = ufos[["centroid-1","centroid-0"]].apply(lambda row: is_point_above_horizon(row, x_centre, y_centre,adj_horizon_r,adj_horizon_l), axis=1)
    ufos=ufos[ufos["area"]>=5]
    ufos=ufos[ufos["area"]<=600]
    ufos=ufos[ufos["axis_major_length"]/ufos["axis_minor_length"]<=5]
    ufos=ufos[ufos["axis_major_length"]>0]
    ufos=ufos[ufos["axis_minor_length"]>0]
    #ufos=ufos[ufos["above_horizon"]==True]
    
    print("7: ufos filtered   "+(str(time.time()-start_time)))
    
    if draw:
        for ufo in range(0,len(ufos)):
            cv2.circle(image, (int(ufos["centroid-1"].iloc[ufo]),int(ufos["centroid-0"].iloc[ufo])), 10, (255,255,255), 3)
        
        #print(image.shape)
        image_with_alpha = cv2.merge([image, horizon_mask[:,:,0]])
        #print(image_with_alpha.shape)
        #cv2.imwrite(outdir+file.replace("jpg", "png"), image_with_alpha)
        cv2.imwrite(outdir+file.replace("jpg", "png"), image)
        
    print("8: ufos drawn and written to png (if draw = True)   "+(str(time.time()-start_time)))
    
    if file==files[0]:
        all_ufos=ufos.copy()
    else:
        all_ufos=pandas.concat([all_ufos, ufos], ignore_index=True)

all_ufos.to_csv(outdir+'detected_ufos.csv', index=False) 

print("9: ufos written to csv   "+(str(time.time()-start_time)))