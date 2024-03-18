# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:41:39 2024

@author: au694732
"""

import pandas as pd,scipy.spatial, json

def compare(file1,file2):

    datasets = {"obs1": pd.read_csv(file1), "obs2": pd.read_csv(file2)}

    for name, dataset in datasets.items():
        obs = dataset.loc[dataset["region_shape_attributes"] != "{}"].copy()
        obs["region_shape_attributes"]=obs["region_shape_attributes"].apply(json.loads)
        attr=pd.json_normalize(obs["region_shape_attributes"].tolist())
        #need to reset the index to ensure they link properly. Watch out in case there 
        #are missing data in some region_attributes column entries...
        attr.reset_index(drop=True, inplace=True)
        obs.reset_index(drop=True, inplace=True)
        #to get rid of 'r' for circle annotations
        attr=attr[["name","cx","cy"]]
        print(attr.columns)
        # Concatenate the original DataFrame with the normalized columns
        obs = pd.concat([obs, attr], axis=1)
        datasets[name] = obs

    print(datasets["obs1"].columns)
    print(set(datasets["obs1"]["filename"]))
    print(set(datasets["obs2"]["filename"]))
    print(len(set(datasets["obs1"]["filename"])))
    print(len(set(datasets["obs2"]["filename"])))
    files=list(set(pd.concat([datasets["obs1"]["filename"],datasets["obs2"]["filename"]])))
    print(len(files))
    
    matchs=[]
    only1s=[]
    only2s=[]
    mismatch=0
    
    for file in files[:]:
        
        print(file)
        matches = {"obs1": [], "obs2": []}
        for name, dataset in datasets.items():
            this_obs=dataset.loc[dataset["filename"]==file].copy()
            other_item = {key: value for key, value in datasets.items() if not key == name}
            other_dataset=next(iter(other_item.values()))
            other_obs=other_dataset.loc[other_dataset["filename"]==file].copy()
            if this_obs.empty:
                matches[name]=[0,len(other_obs)]
                continue
            if other_obs.empty:
                matches[name]=[0,len(this_obs)]
                continue
            xy1=list(zip(this_obs["cx"],this_obs["cy"]))
            xy2=list(zip(other_obs["cx"],other_obs["cy"]))
            kdtree2 = scipy.spatial.cKDTree(xy2)
            distances, indices = kdtree2.query(xy1)
            this_obs.loc[:,"match_id"]=indices
            this_obs.loc[:,"match_dist"]=distances
            this_obs["match"]=this_obs["match_dist"]<15
            this_obs.loc[this_obs[["match","match_id"]].duplicated(),"match"]=False
            print(len(this_obs))
            print(this_obs[["match_id","match","match_dist"]].sort_values(by=["match_id"]))
            matches[name]=[sum(this_obs["match"]),sum(this_obs["match"]==False)]
        
        print(matches)
        if not matches["obs1"][0]==matches["obs2"][0]:
            mismatch+=1
            # raise ValueError("{}: Somehow different numbers of matching points between the two sets, needs debugging".format(file))

        matchs.append(max(matches["obs1"][0],matches["obs2"][0]))
        only1s.append(matches["obs1"][1])
        only2s.append(matches["obs2"][1])

    matchdf=pd.DataFrame({"file":files,"match":matchs,"only1":only1s,"only2":only2s})
    if mismatch>0:
        print("{} mismatch in the number of matches between obs tables. This can happen because second priorities for matches are not currently checked".format(mismatch))

    return(matchdf)
        
    #     # Query nearest neighbors for each point in this_pred
    #     distances, indices = kdtree_set1.query(p_xy)
        
    #     this_pred.loc[:,"match_id"]=indices
    #     this_pred.loc[:,"match_dist"]=distances
    #     this_pred=this_pred.sort_values(by="match_dist",ascending=True)
    #     this_pred["match"]=this_pred["match_dist"]<15
    #     this_pred.loc[this_pred.duplicated(),"match"]=False
        
    #     # tp=sum(this_pred["match"])
    #     # fp=sum(np.logical_not(this_pred["match"]))
    #     # fn=sum(np.logical_not(this_obs["match"]))
    #     # print(tp,fp,fn)
        
    #     try:
    #         all_pred=pd.concat([all_pred, this_pred], ignore_index=True)
    #         all_obs=pd.concat([all_obs, this_obs], ignore_index=True)
    #     except NameError:
    #         all_pred=this_pred.copy()
    #         all_obs=this_obs.copy()
        
    #     if file==files[egFile]:
    
    #         # Load the image
    #         image = cv2.imread(indir+file)
            
    #         for ufo in range(0,len(this_obs)):
    #             cv2.circle(image, (int(this_obs["cx"].iloc[ufo]),int(this_obs["cy"].iloc[ufo])), 10, (255,255,255), 3)
            
    #         cv2.imwrite(outdir+"ufos_observed.png", image)
            
    #         for ufo in range(0,len(this_pred)):
    #             cv2.circle(image, (int(this_pred["centroid-1"].iloc[ufo]),int(this_pred["centroid-0"].iloc[ufo])), 10, (0,255,0), 3)
                
    #         cv2.imwrite(outdir+"ufos_detected.png", image)
    
    # if 'all_pred' not in locals():
    #     tp=0
    #     fp=0
    # else:
    #     tp=sum(all_pred["match"])
    #     fp=sum(np.logical_not(all_pred["match"]))
    #     all_pred.to_csv(outdir+"detected_ufos_eval.csv")
    
    # if 'all_obs' not in locals():
    #     fn=0
    # else:
    #     fn=sum(np.logical_not(all_obs["match"]))
    #     all_obs.to_csv(outdir+"observed_ufos_eval.csv")
    
    # try:
    #     pre=tp/(tp+fp)
    # except ZeroDivisionError:
    #     pre=0
    #     f1=0
    # try:
    #     rec=tp/(tp+fn)
    # except ZeroDivisionError:
    #     rec=0
    #     f1=0
    # try:
    #     f1=2*(pre*rec)/(pre+rec)
    # except ZeroDivisionError:
    #     f1=0
    # print(tp,fp,fn,pre,rec,f1)
    # return(pre,rec,f1)


pd.set_option('display.max_rows', None)

file1=r"O:\Tech_ECOS-OWF-Screening\Fugle-flagermus-havpattedyr\BIRDS\Ship_BasedSurveys\VerticalRadar\Annotations\RDN\project_20231007_final.csv"
file2=r"O:\Tech_ECOS-OWF-Screening\Fugle-flagermus-havpattedyr\BIRDS\Ship_BasedSurveys\VerticalRadar\Annotations\TEO\071023 via_export_csv_teo.csv"

matchdf=compare(file1,file2)
print(sum(matchdf["match"]),sum(matchdf["only1"]),sum(matchdf["only2"]))
print(sum(matchdf["match"])/(sum(matchdf["match"])+sum(matchdf["only1"])))
print(sum(matchdf["match"])/(sum(matchdf["match"])+sum(matchdf["only2"])))

