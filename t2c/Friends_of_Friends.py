"""
Implementing a basic searching algorithm to identify grouped areas of high
values in an array

Can either be imported and used in a seperate script, or run via command line with an input file and an
output file as arguments. Output file will be a string representation of a list, with no modifications, this can be read
in python using the 'ast' module, with:
sizes_list = ast.literal_eval(sizes_string)
"""

import numpy as np
from itertools import count
#from Queue import Queue
from multiprocessing import Queue


def FoF_search(array, threshold):
    """
    :param array: n-dimensional scalar array to search
    :param threshold: float
    :return: (out_map, size_list) n-dimensional array with points filled with corresponding group number
        (0 if not above threshold), and a 1D list containing group sizes
    """
    def cycle_through_options(coord):    # generator which returns indices next to a given index
        for i in range(len(coord)):
            for j in [-1, 1]:
                new_coordinate = [k for k in coord]
                new_coordinate[i] += j
                yield tuple(new_coordinate)

    out_map = np.zeros(array.shape, dtype=int)   # creates an array with the same shape as the input search array
    possibilities = zip(*np.where(array > threshold))  # creates a list of indices of points above the threshold
    poss_set = set(possibilities)
    
    def recursive_search(point, current_group, currentsize):     # function to calculate group membership of a point
        for testPoint in cycle_through_options(point):
            if testPoint in poss_set and not out_map[testPoint]:
                out_map[testPoint] = current_group
                q.put(testPoint)
                currentsize += 1
        return currentsize
    c = count()
    #c.next()
    next(c)
    size_list = []
    q = Queue()             # initialise a queue
    for p in possibilities:     # start cycling through possible points
        if not out_map[p]:           # if the point has not already been searched,
            group = next(c)        # start a new group number
            out_map[p] = group       # assign the corresponding point in the group map to this group
            q.put(p)                # put the point in the queue
            s = 1                   # s contains the new group size
            while not q.empty():    # cycle till queue is empty
                s = recursive_search(q.get(), group, s)  # search each neighbour recursively
            size_list.append(s)     # add size of group to a list
    return out_map, np.array(size_list)


def gaussian(dx, sig):
    """returns a one dimensional gaussian"""
    return np.exp(-dx**2.0/(2.0*sig**2.0))


def halo3d(x, a, sigma, array_size):
    """
    Returns an array with size arSize^3 and one gaussian distribution with amplitude a and s.d. sigma
    somewhere within that array
    :param x: 3d vector to be the mean of gaussian
    :param a: float amplitude
    :param sigma: float s.d.
    :param array_size: size of array to output
    :return: 3d array as detailed above
    """
    ar = np.zeros(array_size, dtype=float)
    for i in range(array_size[0]):
        for j in range(array_size[1]):
            for k in range(array_size[2]):
                dx = float(reduce(lambda foo, y: foo+y**2, [0, i-x[0], j-x[1], k-x[2]]))
                ar[i, j, k] = a*gaussian(dx, sigma)
    return ar

if __name__ == "__main__":
    import sys
    import c2raytools as c2t
    try:
        infile = sys.arg[1]
        outfile = sys.argv[2]
        x_file = c2t.XfracFile(infile)
        ret, sizes = friend_of_friend_search(x_file.xi, 0.5)
        with open(outfile, 'w') as out_file:
            out_file.write(str(sizes))
    except IndexError as e:
        print("Error: expected an input ionised fraction file and an output file")
