# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:53:55 2024

@author: au694732 + Asger Svenning
"""

import numpy as np, ufofinder

#folder="2023 08 10-16"
folder="Radar Grabs 2023 10 07 - 11"
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

par_minmaxstep=[
    [36,48,1],
    [18,40,1],
    [0,20,2],
    [0,2,1],
    [730,750,10],
    [100,300,50],
    [6,10,1],
    [26,32,1],
    [16,20,1],
    [0,2,1]
    ]

lim=159
draw=False
egFile=999

pars = [hue,hi,mis,mia,maa,mid,lwr,hst,rst,hbf]

# Hill climbing algorithm
def hill_climbing(initial_solution, num_iterations):
    num_params = len(initial_solution)
    current_solution = tuple(initial_solution)
    current_value = ufofinder.train(current_solution,folder,x_centre,y_centre,radius,warp,lim,draw=False,egFile=0)

    tried_pars = {}
    tried_pars.update({current_solution:current_value})
    
    improved_pars = {}
    improved_pars.update({current_solution:current_value})
    
    for idx in range(num_iterations*num_params):
        par_idx = idx % num_params
        print("Combinations tried: {}".format(len(tried_pars)))
        neighbors = generate_line(current_solution, par_idx)
        neighbors = [n for n in neighbors if not n in tried_pars]
        print(neighbors)
        if len(neighbors) == 0:
            break
        neighbors_values = [ufofinder.train(neighbor,folder,x_centre,y_centre,radius,warp,lim,draw=False,egFile=0) for neighbor in neighbors]
        [tried_pars.update({n:val}) for n,val in zip(neighbors,neighbors_values)]
        
        # Select the neighbor with the highest objective function value
        best_neighbor = neighbors[np.argmax(neighbors_values)]
        best_value = neighbors_values[np.argmax(neighbors_values)]

        # If the best neighbor has a higher value, update the current solution
        if best_value > current_value:
            current_solution = best_neighbor
            current_value = best_value
            improved_pars.update({current_solution:current_value})
            print("New best solution F1 {}: {}".format(current_value,current_solution))
        # else:
        #     break  # Terminate if there are no better neighbors

    return improved_pars,tried_pars

def generate_line(solution, idx):
    smin, smax, step = par_minmaxstep[idx]
    line_members = []
    for v in range(smin, smax+1,step):
        member = list(solution)
        member[idx] = v
        line_members.extend([tuple(member)])
    return line_members
print(generate_line(pars, 0))

def generate_neighbors(solution):
    # Implement logic to generate neighboring solutions based on the current solution
    # For simplicity, let's consider incrementing/decrementing each variable by 1
    neighbors = []
    for i in range(len(solution)):
        neighbor_plus = list(solution)
        neighbor_minus = list(solution)
        neighbor_plus[i] += 1
        neighbor_minus[i] -= 1
        neighbors.extend([tuple(neighbor_plus), tuple(neighbor_minus)])
    return neighbors
#print(generate_neighbors(pars))

improved_pars,tried_pars=hill_climbing(pars, 5)
print(tried_pars)
print(improved_pars)
#print(ufofinder.train(pars,folder,x_centre,y_centre,radius,warp,lim=2,draw=False,egFile=0))
#print(minimize(ufofinder.train, pars, args=(folder,x_centre,y_centre,radius,warp,lim,draw,egFile), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options={'maxiter': 2}))
