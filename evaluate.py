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

# List all files in the directory ending with ".jpg"
jpg_files = [f for f in os.listdir(indir) if f.lower().endswith(".jpg")]

files=sorted(jpg_files)
file=files[31]

pred=pd.read_csv(outdir+"detected_ufos.csv")
pred=pred[pred["file"]==file]
print(pred)

obs=pd.read_csv(r"O:\Tech_ECOS-OWF-Screening\Fugle-flagermus-havpattedyr\BIRDS\Ship_BasedSurveys\VerticalRadar\Annotations\RDN\project_20231007_final.csv")
#obs = obs.drop('name', axis=1)

obs=obs[obs["filename"]==file]
print(obs["region_shape_attributes"])
print(obs.columns)

#below is to unserialize the dictionary column
obs["region_shape_attributes"]=obs["region_shape_attributes"].apply(json.loads)
print(obs.columns)

attr=pd.json_normalize(obs["region_shape_attributes"].tolist())
#need to reset the index to ensure they link properly. Watch out in case there 
#are missing data in some region_attributes column entries...
attr.reset_index(drop=True, inplace=True)
obs.reset_index(drop=True, inplace=True)
print(attr.columns)
print(attr)

# Concatenate the original DataFrame with the normalized columns
obs = pd.concat([obs, attr], axis=1)

# # Drop the original 'Details' column if needed
# obs = obs.drop('region_shape_attributes', axis=1)

print(obs)

o_xy=list(zip(obs["cx"],obs["cy"]))
p_xy=list(zip(pred["centroid-1"],pred["centroid-0"]))

print(o_xy)
print(p_xy)

# Build KD trees for both sets of points
kdtree_set1 = scipy.spatial.cKDTree(o_xy)
kdtree_set2 = scipy.spatial.cKDTree(p_xy)

# Query nearest neighbors for each point in obs
distances, indices = kdtree_set2.query(o_xy)

obs["match_id"]=indices
obs["match_dist"]=distances
obs=obs.sort_values(by="match_dist",ascending=True)
obs["match"]=obs["match_dist"]<15
obs.loc[obs.duplicated(subset=[x for x in obs.columns if x != "region_shape_attributes"]),"match"]=False
obs.to_csv(outdir+"output_file_obs.csv")

# Query nearest neighbors for each point in pred
distances, indices = kdtree_set1.query(p_xy)

pred["match_id"]=indices
pred["match_dist"]=distances
pred=pred.sort_values(by="match_dist",ascending=True)
pred["match"]=pred["match_dist"]<15
pred.loc[pred.duplicated(),"match"]=False
pred.to_csv(outdir+"output_file_pred.csv")

tp=sum(pred["match"])
fp=sum(np.logical_not(pred["match"]))
fn=sum(np.logical_not(obs["match"]))
print(tp,fp,fn)

pre=tp/(tp+fp)
rec=tp/(tp+fn)
f1=2*(pre*rec)/(pre+rec)
print(pre,rec,f1)


# Load the image
image = cv2.imread(indir+file)

for ufo in range(0,len(obs)):
    cv2.circle(image, (int(obs["cx"].iloc[ufo]),int(obs["cy"].iloc[ufo])), 10, (255,255,255), 3)

cv2.imwrite(outdir+"ufos_observed.png", image)

for ufo in range(0,len(pred)):
    if pred.loc[ufo,"match"]:
        cv2.circle(image, (int(pred["centroid-1"].iloc[ufo]),int(pred["centroid-0"].iloc[ufo])), 10, (0,255,0), 3)
    
cv2.imwrite(outdir+"ufos_detected.png", image)