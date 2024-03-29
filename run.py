# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:13:45 2024

@author: au694732
"""

import numpy as np, ufofinder

folder="2023 08 10-16"
#folder="Radar Grabs 2023 10 07 - 11"
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
hue=37
hi=15

draw=True
lim=150
egFile=149

pars = [hue,hi,mis,mia,maa,mid,lwr,hst,rst,hbf]

ufofinder.predict(pars,folder,x_centre,y_centre,radius,warp,lim=lim,draw=draw,egFile=egFile)