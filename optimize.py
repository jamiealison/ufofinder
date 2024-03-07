# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:53:55 2024

@author: au694732
"""

import ufofinder

#folder="2023 08 10-16"
folder="Radar Grabs 2023 10 07 - 11"
x_centre=990
y_centre=950
radius=929
warp=0.989724175229854
hst=20
rst=20
hbf=2
mis=60
mia=5
maa=600
mid=50
lwr=5
hue=30
hi=5

print(ufofinder.train(folder,hue,hi,mis,mia,maa,mid,lwr,hst,rst,hbf,x_centre,y_centre,radius,warp,lim=2,draw=False,egFile=0))