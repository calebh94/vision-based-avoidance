# -*- coding: utf-8 -*-
"""
Created on Wed May  2 11:09:51 2018

@author: jstickney7
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Node class
class Node:
    def __init__(self, value, point):
        self.value = value
        self.point = point
        self.parent = None
        self.H = 0
        self.G = 0

    # Manhattan Distance
    def Hval(self, point):
        return np.abs(self.point[0] - point[0]) + np.abs(self.point[1] - point[1]) + np.abs(self.point[2]-point[2])

    def move_cost(self, point):
        return np.sqrt((self.point[0] - point[0]) ** 2 + (self.point[1] - point[1]) ** 2 + (self.point[2]-point[2])**2)


def children(point, grid):
    x, y, z = point.point
    links = [grid[d[0]][d[1]][d[2]] for d in
             [(x - 1, y, z), (x, y - 1, z), (x, y + 1, z), (x + 1, y, z), (x + 1, y + 1, z), (x + 1, y - 1, z), (x - 1, y - 1, z),
              (x - 1, y + 1, z),
              (x - 1, y, z-1), (x, y - 1, z-1), (x, y + 1, z-1), (x + 1, y, z-1), (x + 1, y + 1, z-1), (x + 1, y - 1, z-1),
              (x - 1, y - 1, z-1),
              (x - 1, y, z + 1), (x, y - 1, z + 1), (x, y + 1, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1),
              (x + 1, y - 1, z + 1),
              ]]
    return [link for link in links if link.value != '%']


# Astar Algorithm
def astar(start, goal, grid, walls):
    # Create open and closed lists
    openlst = []
    closelst = []

    # gnerate all nodes
    world = [[[[] for j in range(grid)] for i in range(grid)] for x in range(grid)]
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                world[i][j][k] = Node(0, (i, j, k))

    # Evaluate Manhattan Distance to goal
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                node = world[i][j][k]
                node.H = node.Hval(goal)

    # Add start point to open lst
    current = world[start[0]][start[1]][start[2]]
    openlst.append(current)

    # Search
    while openlst:

        # find item with lowest score
        current = min(openlst, key=lambda o: o.G + o.H)

        # If we are at goal retrace path
        if current.point == goal:
            path = []
            while current.parent:
                path.append(current)
                current = current.parent
            path.append(current)
            lst = []
            for i in range(len(openlst)):
                lst.append(openlst[i].point)
            return path, lst

        # Remove current position from open list
        openlst.remove(current)

        # Add current position to closed list
        closelst.append(current)

        # Generate children of current spot
        for node in children(current, world):
            if node in closelst:
                continue
            if node in openlst:
                # check if we beat the G score
                new_g = current.G + current.move_cost(node.point)
                if node.G > new_g:
                    node.G = new_g
                    node.parent = current
            else:

                # Determine if point is blocked or not
                if node.point in walls:
                    continue
                else:
                    # if it isn't in the open set, calculate score
                    node.G = current.G + current.move_cost(node.point)

                    # assign Parent node
                    node.parent = current

                    # Add to openlst
                    openlst.append(node)

    raise ValueError('No Path Found')


# Generate initial values
grid = 20
start, goal = (2, 4,0), (13, 8, 8)

# Define Walls
walls = [(1, 5,4), (1, 6,4), (1, 7,4), (4, 5, 4), (4, 4, 4), (4, 3, 5), (4, 2, 5), (4, 6, 5), (6, 6, 5), (6, 7, 5), (6, 8, 5), (6, 9, 5), (6, 10, 5),
         (9, 8, 5), (9, 9, 5)]

# Run A_star program
take, opl = astar(start, goal, grid, walls)

nodes = []
for i in range(len(take)):
    nodes.append(take[i].point)

print(nodes)
a, b, c = zip(*nodes)
a, b, c = list(a), list(b), list(c)
wa, wb, wc = zip(*walls)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(a,b,c)
ax.scatter(wa, wb, wc, c='r', marker='X')
ax.scatter(goal[0],goal[1],goal[2],c='g', marker = '^')
plt.show()
