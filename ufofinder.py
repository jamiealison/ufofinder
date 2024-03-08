# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:27:34 2024

@author: au694732
"""

import numpy as np, math, cv2, os, skimage.measure,pandas as pd,time,scipy.spatial, json

def predict(pars,folder,x_centre,y_centre,radius,warp,lim=101,draw=False,egFile=0):
    
    print("Training {}".format(pars))
    hue,hi,mis,mia,maa,mid,lwr,hst,rst,hbf=pars
    indir="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/ScreenDumps/"+folder+"/"
    outdir="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/Predictions/"+folder+"/"
    #set ellipse radius
    radius2=int(radius*warp)
    
    # Record start time
    start_time = time.time()
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # List all files in the directory ending with ".jpg"
    jpg_files = [f for f in os.listdir(indir) if f.lower().endswith(".jpg")]
    
    files=sorted(jpg_files)
    
    print("1: initial setup:   "+(str(time.time()-start_time)))
    
    if len(files)<lim:
        raise ValueError("lim is higher then the number of jpgs in the directory")
    
    for file in files[:lim]:
    
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
        
        s_below_min=s_cone<mis
        h[s_below_min]=0
        
        # h_sm=scipy.ndimage.uniform_filter(h,size=3)
        # cv2.imwrite(wd+"an_output_image_r_sm.png", h_sm)
        
        h_th=np.logical_and(h>=hue-hi,h<=hue+hi).astype(int) * 255
        if file==files[egFile]:
            cv2.imwrite(outdir+"an_output_image_relevant.png", h_th)
         
        yellownesses=[]
        #angles=[90]
        angles=[x / 2.0 for x in range(0, 720)]
        
        print("2: image loaded and colorspace converted   "+(str(time.time()-start_time)))
        
        for angle in angles:
        
            # Define a line (start and end points)
            line=line_ends_from_angle(x_centre, y_centre, radius, angle, warp)
            
            # Get pixels that intersect the line
            intersecting_pixels = pixels_on_line(h_th, line)
            
            # Convert the list of coordinates to a NumPy array for indexing
            coordinates_array = np.array(intersecting_pixels)
            
            # Extract x and y coordinates separately
            x_coords, y_coords = coordinates_array[:, 0], coordinates_array[:, 1]
            
            yellowness=np.mean(h_th[y_coords, x_coords])
            
            # below is to plot the horizon lines
            if(yellowness>hst and file==files[egFile]):
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
        #print(horizon_r)
        #print(horizon_l)
        
        yellownesses=np.array(yellownesses).astype(int)
        #print(yellownesses)
        
        horizons = pd.DataFrame({'angle': angles, 'strength': yellownesses})
        artefact_horizons=horizons.loc[horizons['strength']>rst,['angle']]
        artefact_horizons=artefact_horizons['angle'].tolist()
        #print(artefact_horizons)
        
        horizon_r_clean=angles[np.where(yellownesses>hst)[0][0]]-hbf/2 if np.any(yellownesses>hst) else 135
        horizon_l_clean=angles[np.where(yellownesses>hst)[0][-1]]+hbf/2 if np.any(yellownesses>hst) else 225
        #print(horizon_r_clean)
        #print(horizon_l_clean)
        #here we use an artificial horizon to extract bad weather conditions
        
        adj_horizon_r=horizon_r_clean
        adj_horizon_l=horizon_l_clean
        #adj_horizon_r=70
        #adj_horizon_l=290
        
        circle_mask = np.zeros_like(image)
        # cv2.circle(circle_mask, (x_centre, y_centre), radius, (255, 255, 255), thickness=-1)
        cv2.ellipse(circle_mask, (x_centre, y_centre), (radius,radius2), 0, 0, 360, (255, 255, 255), thickness=-1)
        
        image_masked=np.copy(image)
        image_masked[circle_mask[:,:,0]==0,:]=0
        
        print("4: circle mask generated   "+(str(time.time()-start_time)))
        
        horizon_mask=np.zeros_like(image)
        #-90 to the start angle as the function thinks 0 is east. +270 to the second angle to add 360 while accounting for the east problem. Second angle must be larger than the first as drawing is always clockwise.
        cv2.ellipse(horizon_mask, (x_centre, y_centre), (radius,radius2), 0, adj_horizon_l-90, adj_horizon_r+270, (255, 255, 255), thickness=-1)
        
        print("5: horizon mask generated   "+(str(time.time()-start_time)))
        
        #important - we remove the areas outside the from the hue mask, but not the areas below the horizon
        h_th[circle_mask[:,:,0]==0,]=0
        
        if file==files[egFile]:
            cv2.imwrite(outdir+"an_output_image_segments.png", h_th)
            # Save the image as JPEG
            cv2.imwrite(outdir+"an_output_image_masked.png", image_masked)
            # Save the image as JPEG
            cv2.imwrite(outdir+"an_output_image_horizons.png", image_angles)
            cv2.imwrite(outdir+"an_output_image_horizon_mask.png", horizon_mask)
        
        #below may be useful when finding bad weather
        #valid=np.transpose(np.where(horizon_mask[:,:,0] > 0))
        #valid_r=np.percentile(image_masked[valid[:,0],valid[:,1],2], 95)
        #print(valid_r)
        
    
        image_ufos,n_ufos=skimage.measure.label(h_th,return_num=True)
        print("6: UFOs detected before filtering: {},     {}".format(n_ufos,str(time.time()-start_time)))
        
        ufos_dict=skimage.measure.regionprops_table(image_ufos, properties=('label','centroid','bbox','area','axis_major_length','axis_minor_length'))
        ufos=pd.DataFrame(ufos_dict)
        
        dx = abs(x_centre - ufos["centroid-1"])
        dy = abs(y_centre - ufos["centroid-0"])
        d = np.sqrt(dx**2+(dy/warp)**2)
        #note the distance is in strange units - "WARPED PIXELS"
        ufos=ufos.assign(distance=d)
        
        #filters
        #best do removal of close-by detections first
        out=d>mid
        ufos=ufos[out]
        ufos["file"]=file
        ufos["horizon_l"]=horizon_l
        ufos["horizon_r"]=horizon_r
        pix_x=ufos["centroid-1"].astype(int).tolist()
        pix_y=ufos["centroid-0"].astype(int).tolist()
        ufos["above_horizon"]=horizon_mask[pix_y, pix_x,0] == 255
        #below an alternative way to calculate whether ufos are above horizon, but horizon mask is now so fast that it's not needed
        #ufos['above_line'] = ufos[["centroid-1","centroid-0"]].apply(lambda row: is_point_above_horizon(row, x_centre, y_centre,adj_horizon_r,adj_horizon_l), axis=1)
        ufos['angle'] = ufos[["centroid-1","centroid-0"]].apply(lambda row: angle_of_point(row, x_centre, y_centre), axis=1)
        ufos["radial_artefact"]=ufos["angle"].apply(lambda row: angle_in_artefacts(row, artefact_horizons))
        #ensure distinction between raidal artefacts and below horizon
        ufos.loc[ufos["above_horizon"]==False,["radial_artefact"]]=False
        ufos=ufos[ufos["area"]>=mia]
        ufos=ufos[ufos["area"]<=maa]
        ufos=ufos[ufos["axis_major_length"]/ufos["axis_minor_length"]<=lwr]
        ufos=ufos[ufos["axis_major_length"]>0]
        ufos=ufos[ufos["axis_minor_length"]>0]
        ufos=ufos[ufos["above_horizon"]==True]
        ufos=ufos[ufos["radial_artefact"]==False]
        
        print("7: ufos filtered   "+(str(time.time()-start_time)))
        
        if draw:
            for ufo in range(0,len(ufos)):
                cv2.circle(image, (int(ufos["centroid-1"].iloc[ufo]),int(ufos["centroid-0"].iloc[ufo])), 10, (255,255,255), 3)
            
            #print(image.shape)
            #image_with_alpha = cv2.merge([image, horizon_mask[:,:,0]])
            #print(image_with_alpha.shape)
            #cv2.imwrite(outdir+file.replace("jpg", "png"), image_with_alpha)
            cv2.imwrite(outdir+file.replace("jpg", "png"), image)
            
        print("8: ufos drawn and written to png (if draw = True)   "+(str(time.time()-start_time)))
        
        if file==files[0]:
            all_ufos=ufos.copy()
        else:
            all_ufos=pd.concat([all_ufos, ufos], ignore_index=True)
    
    all_ufos.to_csv(outdir+'detected_ufos.csv', index=False) 
    
    print("9: ufos written to csv   "+(str(time.time()-start_time)))

def evaluate(folder,lim=101,egFile=0):

    indir="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/ScreenDumps/"+folder+"/"
    outdir="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/Predictions/"+folder+"/"
    
    pred=pd.read_csv(outdir+"detected_ufos.csv")
    
    obs=pd.read_csv(r"O:\Tech_ECOS-OWF-Screening\Fugle-flagermus-havpattedyr\BIRDS\Ship_BasedSurveys\VerticalRadar\Annotations\RDN\project_20231007_final.csv")
    #obs = obs.drop('name', axis=1)
    
    obs = obs.loc[obs["region_shape_attributes"] != "{}"]
    
    #below is to unserialize the dictionary column
    obs["region_shape_attributes"]=obs["region_shape_attributes"].apply(json.loads)
    #print(obs.columns)
    
    attr=pd.json_normalize(obs["region_shape_attributes"].tolist())
    #need to reset the index to ensure they link properly. Watch out in case there 
    #are missing data in some region_attributes column entries...
    attr.reset_index(drop=True, inplace=True)
    obs.reset_index(drop=True, inplace=True)
    #print(attr.columns)
    
    # List all files in the directory ending with ".jpg"
    jpg_files = [f for f in os.listdir(indir) if f.lower().endswith(".jpg")]
    
    files=sorted(jpg_files)
    
    # Concatenate the original DataFrame with the normalized columns
    obs = pd.concat([obs, attr], axis=1)
    
    for file in files[:lim]:
    
        this_pred=pred.loc[pred["file"]==file].copy()
        this_obs=obs.loc[obs["filename"]==file].copy()
        
        #WARNING
        #currently skipping files where either dframe is empty for convenience
        if this_pred.empty or this_obs.empty:
            continue
    
        # # Drop the original 'Details' column if needed
        # obs = obs.drop('region_shape_attributes', axis=1)
        
        o_xy=list(zip(this_obs["cx"],this_obs["cy"]))
        p_xy=list(zip(this_pred["centroid-1"],this_pred["centroid-0"]))
        
        print(file)
        # Build KD trees for both sets of points
        kdtree_set1 = scipy.spatial.cKDTree(o_xy)
        kdtree_set2 = scipy.spatial.cKDTree(p_xy)
        
        # Query nearest neighbors for each point in this_obs
        distances, indices = kdtree_set2.query(o_xy)
        
        this_obs.loc[:,"match_id"]=indices
        this_obs.loc[:,"match_dist"]=distances
        this_obs=this_obs.sort_values(by="match_dist",ascending=True)
        this_obs["match"]=this_obs["match_dist"]<15
        this_obs.loc[this_obs.duplicated(subset=[x for x in this_obs.columns if x != "region_shape_attributes"]),"match"]=False
        
        # Query nearest neighbors for each point in this_pred
        distances, indices = kdtree_set1.query(p_xy)
        
        this_pred.loc[:,"match_id"]=indices
        this_pred.loc[:,"match_dist"]=distances
        this_pred=this_pred.sort_values(by="match_dist",ascending=True)
        this_pred["match"]=this_pred["match_dist"]<15
        this_pred.loc[this_pred.duplicated(),"match"]=False
        
        # tp=sum(this_pred["match"])
        # fp=sum(np.logical_not(this_pred["match"]))
        # fn=sum(np.logical_not(this_obs["match"]))
        # print(tp,fp,fn)
        
        try:
            all_pred=pd.concat([all_pred, this_pred], ignore_index=True)
            all_obs=pd.concat([all_obs, this_obs], ignore_index=True)
        except NameError:
            all_pred=this_pred.copy()
            all_obs=this_obs.copy()
        
        if file==files[egFile]:
    
            # Load the image
            image = cv2.imread(indir+file)
            
            for ufo in range(0,len(this_obs)):
                cv2.circle(image, (int(this_obs["cx"].iloc[ufo]),int(this_obs["cy"].iloc[ufo])), 10, (255,255,255), 3)
            
            cv2.imwrite(outdir+"ufos_observed.png", image)
            
            for ufo in range(0,len(this_pred)):
                cv2.circle(image, (int(this_pred["centroid-1"].iloc[ufo]),int(this_pred["centroid-0"].iloc[ufo])), 10, (0,255,0), 3)
                
            cv2.imwrite(outdir+"ufos_detected.png", image)
    
    if 'all_pred' not in locals():
        tp=0
        fp=0
    else:
        tp=sum(all_pred["match"])
        fp=sum(np.logical_not(all_pred["match"]))
        all_pred.to_csv(outdir+"detected_ufos_eval.csv")
    
    if 'all_obs' not in locals():
        fn=0
    else:
        fn=sum(np.logical_not(all_obs["match"]))
        all_obs.to_csv(outdir+"observed_ufos_eval.csv")
    
    try:
        pre=tp/(tp+fp)
    except ZeroDivisionError:
        pre=0
        f1=0
    try:
        rec=tp/(tp+fn)
    except ZeroDivisionError:
        rec=0
        f1=0
    try:
        f1=2*(pre*rec)/(pre+rec)
    except ZeroDivisionError:
        f1=0
    print(tp,fp,fn,pre,rec,f1)
    return(pre,rec,f1)

def train(pars,folder,x_centre,y_centre,radius,warp,lim=101,draw=False,egFile=0):
    
    predict(pars,folder,x_centre,y_centre,radius,warp,lim,draw,egFile)
    return(evaluate(folder,lim=lim)[2])

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

def angle_of_point(point, x_centre, y_centre):
    x, y = point
    # Calculate the angle in radians using atan2
    angle_rad = math.atan2(y_centre-y, x_centre-x)
    # Convert the angle to degrees and shift it so that 0 degrees is north
    angle_deg = (math.degrees(angle_rad)-90) % 360
    
    return angle_deg

def angle_in_artefacts(angle,artefacts):
    diffs=[angle - artefact for artefact in artefacts]
    abs_diffs=[abs(x) for x in diffs]
    small_diffs=[x<0.5 for x in abs_diffs]
    return any(small_diffs)
