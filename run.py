# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:13:45 2024

@author: au694732
"""

import ufofinder

#folder="2023 06 13-18"
#folder="2023 08 10-16"
#folder="Radar Grabs 2023 09 15 -23"
#folder="Radar Grabs 2023 10 07 - 11"
folder="Radar Grabs 2023 12 02-08"
x_centre=990
y_centre=950
radius=929
warp=0.989724175229854
hst=28
rst=18
hbf=0
mis=0
mia=0
maa=740
mid=100
lwr=8
hue=42
hi=20

draw=True
lim=0
egFile=100

pars = [hue,hi,mis,mia,maa,mid,lwr,hst,rst,hbf]

ufofinder.predict(pars,folder,x_centre,y_centre,radius,warp,lim=lim,draw=draw,egFile=egFile)