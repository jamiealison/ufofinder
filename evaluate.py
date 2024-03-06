# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:04:56 2024

@author: au694732
"""

import numpy as np, miniball, math, cv2, scipy.ndimage,scipy.spatial, os, statistics,skimage.measure,pandas as pd, json

#folder="2023 08 10-16"
folder="Radar Grabs 2023 10 07 - 11"

indir="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/ScreenDumps/"+folder+"/"
outdir="O:/Tech_ECOS-OWF-Screening/Fugle-flagermus-havpattedyr/BIRDS/Ship_BasedSurveys/VerticalRadar/Predictions/"+folder+"/"

pred=pd.read_csv(outdir+"detected_ufos.csv")

obs=pd.read_csv(r"O:\Tech_ECOS-OWF-Screening\Fugle-flagermus-havpattedyr\BIRDS\Ship_BasedSurveys\VerticalRadar\Annotations\RDN\project_20231007_final.csv")
#obs = obs.drop('name', axis=1)

obs = obs.loc[obs["region_shape_attributes"] != "{}"]

#below is to unserialize the dictionary column
obs["region_shape_attributes"]=obs["region_shape_attributes"].apply(json.loads)
print(obs.columns)

attr=pd.json_normalize(obs["region_shape_attributes"].tolist())
#need to reset the index to ensure they link properly. Watch out in case there 
#are missing data in some region_attributes column entries...
attr.reset_index(drop=True, inplace=True)
obs.reset_index(drop=True, inplace=True)
print(attr.columns)

# List all files in the directory ending with ".jpg"
jpg_files = [f for f in os.listdir(indir) if f.lower().endswith(".jpg")]

files=sorted(jpg_files)

# Concatenate the original DataFrame with the normalized columns
obs = pd.concat([obs, attr], axis=1)


for file in files[:101]:

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
    
    if file==files[31]:

        # Load the image
        image = cv2.imread(indir+file)
        
        for ufo in range(0,len(this_obs)):
            cv2.circle(image, (int(this_obs["cx"].iloc[ufo]),int(this_obs["cy"].iloc[ufo])), 10, (255,255,255), 3)
        
        cv2.imwrite(outdir+"ufos_observed.png", image)
        
        for ufo in range(0,len(this_pred)):
            cv2.circle(image, (int(this_pred["centroid-1"].iloc[ufo]),int(this_pred["centroid-0"].iloc[ufo])), 10, (0,255,0), 3)
            
        cv2.imwrite(outdir+"ufos_detected.png", image)

all_obs.to_csv(outdir+"observed_ufos_eval.csv")
all_pred.to_csv(outdir+"detected_ufos_eval.csv")

tp=sum(all_pred["match"])
fp=sum(np.logical_not(all_pred["match"]))
fn=sum(np.logical_not(all_obs["match"]))
print(tp,fp,fn)

pre=tp/(tp+fp)
rec=tp/(tp+fn)
f1=2*(pre*rec)/(pre+rec)
print(pre,rec,f1)

