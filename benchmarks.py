"""
https://www.sfu.ca/~ssurjano/optimization.html
"""
import numpy as np
from functools import reduce
def eggholder(sol):
    """Eggholder function

    Description:
    - Dimensions: 2
    - The Eggholder function is a difficult function to optimize, because of the large number of local minima.

    Input Domain:
    - The function is usually evaluated on the square xi ∈ [-512, 512], for all i = 1, 2.
    """
    result = -(sol[1] + 47)*np.sin( np.sqrt(np.abs(sol[1] + sol[0]/2 + 47)) ) - sol[0]*np.sin( np.sqrt(np.abs(sol[0] - sol[1] + 47)) )
    return result

def w(x):
    return 1 + (x-1)/4

def levy(sol):
    dim = len(sol)
    wsol = [w(x) for x in sol]
    som = reduce(lambda acc,e:acc+(e-1)**2 * (1 + 10*np.sin(np.pi * e + 1)**2),wsol,0)
    return np.sin(np.pi * wsol[0])**2 + som + (wsol[dim-1]-1)**2 * (1 + np.sin(2*np.pi * wsol[dim-1])**2)

def sphere(sol):
    return reduce(lambda acc,e:acc+e*e,sol,0)

def griewank(sol):
    (s,p,i) = reduce(lambda acc,e:(acc[0]+e*e,acc[1]*np.cos(e/np.sqrt(acc[2])),acc[2]+1),sol,(0,1,1))
    return s/4000


def get_evaluation_by_name(f_name):
    if f_name == "sphere":
        return sphere
    elif f_name == "griewank":
        return griewank
    elif f_name == "levy":
        return levy
    elif f_name == "eggholder":
        return eggholder
    else:
        print("error!")
        exit(1)