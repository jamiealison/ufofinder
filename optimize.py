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
hst=20
rst=20
hbf=4
mis=60
mia=5
maa=600
mid=50
lwr=5
hue=30
hi=5

par_minmax=[
    [25,34],
    [0,9],
    [0,100],
    [0,9],
    [300,600],
    [50,100],
    [0,9],
    [10,30],
    [0,30],
    [0,9]
    ]

lim=101
draw=False
egFile=999

pars = [hue,hi,mis,mia,maa,mid,lwr,hst,rst,hbf]

# Hill climbing algorithm
def hill_climbing(initial_solution, num_iterations):
    num_params = len(initial_solution)
    current_solution = tuple(initial_solution)
    current_value = ufofinder.train(current_solution,folder,x_centre,y_centre,radius,warp,lim,draw=False,egFile=0)

    tried_pars = set()
    tried_pars.add(current_solution)
    
    improved_pars = set()
    improved_pars.add(current_solution)
    
    for idx in range(num_iterations*num_params):
        par_idx = idx % num_params
        print("Combinations tried: {}".format(len(tried_pars)))
        neighbors = generate_line(current_solution, par_idx)
        neighbors = [n for n in neighbors if not n in tried_pars]
        print(neighbors)
        if len(neighbors) == 0:
            break
        neighbors_values = [ufofinder.train(neighbor,folder,x_centre,y_centre,radius,warp,lim,draw=False,egFile=0) for neighbor in neighbors]
        [tried_pars.add(n) for n in neighbors]
        
        # Select the neighbor with the highest objective function value
        best_neighbor = neighbors[np.argmax(neighbors_values)]
        best_value = neighbors_values[np.argmax(neighbors_values)]

        # If the best neighbor has a higher value, update the current solution
        if best_value > current_value:
            current_solution = best_neighbor
            current_value = best_value
            improved_pars.add(best_neighbor)
            print("New best solution F1 {}: {}".format(current_value,current_solution))
        else:
            break  # Terminate if there are no better neighbors

    return improved_pars

def generate_line(solution, idx, linewidth=4):
    smin, smax = par_minmax[idx]
    current = solution[idx]
    smin = max(current-linewidth,smin)
    smax = min(current+linewidth,smax)
    line_members = []
    for v in range(smin, smax+1):
        member = list(solution)
        member[idx] = v
        line_members.extend([tuple(member)])
    return line_members
print(generate_line(pars, 0, 1))

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

print(hill_climbing(pars, 5))
#print(ufofinder.train(pars,folder,x_centre,y_centre,radius,warp,lim=2,draw=False,egFile=0))
#print(minimize(ufofinder.train, pars, args=(folder,x_centre,y_centre,radius,warp,lim,draw,egFile), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options={'maxiter': 2}))
