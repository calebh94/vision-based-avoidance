#!/usr/bin/env python

'''

Astar function for obstacle avoidance using 2D occupancy grid from Octomap_server

@author:  Caleb Harris <charris92@gatech.edu>

'''


import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, value, point, isOccupied):
        self.value = value
        self.point = point
        self.global_point = (None, None)
        self.isOccupied = isOccupied
        self.costToGoal = 0
        self.costToGo = 0
        self.parent = None

        if len(point) is not 2:
            ValueError('Point input must be in 2D!')

    def calculateCostToGoal(self, point):
        dis = np.abs(self.point[0] - point[0]) + np.abs(self.point[1] - point[1])
        # dis = np.sqrt((self.point[0] - point[0]) ** 2 + (self.point[1] - point[1]) ** 2)
        self.costToGoal = dis

    def calculateMoveCost(self, point):
        cost = np.sqrt((self.point[0] - point[0]) ** 2 + (self.point[1] - point[1]) ** 2)
        self.costToGo = cost


def children(point, world):
    x, y = point.point
    links = []
    neighbors = [(x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y - 1),
      (x - 1, y + 1),
      ]
    for d in neighbors:
        if d[0] >= np.size(world,0) or d[1] >= np.size(world,1):
            continue
        elif d[0] < 0 or d[1] < 0:
            continue
        if world[d[0]][d[1]] is not None:
            links.append(world[d[0]][d[1]])

    return links


class Astar2D:

    def __init__(self, namespace='DJI', method='expert', input_data='occupancy_grid', debug=False):
        self.namespace = namespace
        self.method = method
        self.input_data = input_data
        self.debug = debug

        self.start = []
        self.goal = []
        self.graph = []
        self.path = []
        self.nextpoint = []
        self.obstacles = []


    def run(self, start, goal, graph):
        path, lst, obstacles = self.find_path(start, goal, graph)
        if len(path) == 0:
            path = self.path
            obstacles = self.obstacles
        else:
            nodes = []
            walls = []
            allpoints = []
            allobstacles = []

            for i in range(len(path)):
                # nodes.append(path[i].global_point + 0.5)
                nodes.append((path[i].global_point[0] + 0.5, path[i].global_point[1] + 0.5))

            # nodes.append((0,0))
            # nodes.insert(0,(5,0))

            for j in range(len(obstacles)):
                walls.append((obstacles[j].global_point[0] + 0.5, obstacles[j].global_point[1] + 0.5))

            print(nodes)
            a, b = zip(*nodes)
            # a, b = list(a), list(b)
            print(walls)

            if len(walls) != 0:
                wa, wb = zip(*walls)
            else:
                wa, wb = (None, None)
            # Plot the full zone
            for i in range(0, np.size(graph.cells, 0)):
                for j in range(0, np.size(graph.cells, 1)):
                    if graph.cells[i][j].occupied:
                        allobstacles.append(
                            (graph.cells[i][j].x + graph.origin[0] + 0.5, graph.cells[i][j].y + graph.origin[1] + 0.5))
                    else:
                        allpoints.append(
                            (graph.cells[i][j].x + graph.origin[0] + 0.5, graph.cells[i][j].y + graph.origin[1] + 0.5))
            all_x, all_y = zip(*allpoints)
            wall_x, wall_y = zip(*allobstacles)

            if len(allobstacles) == 0:
                wall_x, wall_y = [],[]
            else:
                wall_x, wall_y = zip(*allobstacles)

            self.path = nodes
            self.obstacles = obstacles
            self.nextpoint = [self.path[-2][0], self.path[-2][1]]

            if self.debug:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(a, b, c='g')

                ax.scatter(all_x, all_y, c='b', marker='o')
                ax.scatter(wall_x, wall_y, c='r', marker='s')

                ax.scatter(wa, wb, c='r', marker='X')
                ax.scatter(start[0], start[1], c='g', marker='^')
                ax.scatter(goal[0], goal[1], c='g', marker='*')

                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                ax2.scatter(all_x, all_y, c='b', marker='o')
                ax2.scatter(wall_x, wall_y, c='r', marker='s')
                plt.show()






    def find_path(self, start, goal, graph):
        self.start = start
        self.goal = goal
        self.graph = graph

        grid = self.graph

        # round start and goal
        start = (round(start[0],1), round(start[1],1))
        goal = (round(goal[0],1), round(goal[1],1))
        # goal = (0,1)


        # Create open and closed lists
        openlst = []
        closelst = []

        nodes = [[[] for i in range(np.size(grid.cells,1))] for j in range(np.size(grid.cells,0))]
        obstacles = []

        for i in range(0,np.size(grid.cells,0)):
            for j in range(0,np.size(grid.cells,1)):
                nodes[i][j] = Node(0, (int(grid.cells[i][j].x), int(grid.cells[i][j].y)), grid.cells[i][j].occupied)

        # Update node global values with references from grid
        for i in range(0,np.size(grid.cells,0)):
            for j in range(0,np.size(grid.cells,1)):
                new_x = nodes[i][j].point[0] * grid.mpn + grid.origin[0]
                new_y = nodes[i][j].point[1] * grid.mpn + grid.origin[1]
                nodes[i][j].global_point = (new_x, new_y)
                # nodes[i][j].point[0] = nodes[i][j].point[0] * grid.mpn + grid.origin[0]
                # nodes[i][j].point[1] = nodes[i][j].point[1] * grid.mpn + grid.origin[1]

        # Add start point to open lst
        start_index = None
        goal_index = None
        for i in range(0, np.size(grid.cells, 0)):
            for j in range(0, np.size(grid.cells, 1)):
                if abs(float(nodes[i][j].global_point[0]) - float(start[0])) <= 0.5 and abs(
                        float(nodes[i][j].global_point[1]) - float(start[1])) <= 0.5:
                    start_index = (i, j)
                elif abs(float(nodes[i][j].global_point[0]) - float(goal[0])) <= 0.5 and abs(
                        float(nodes[i][j].global_point[1]) - float(goal[1])) <= 0.5:
                    goal_index = (i, j)
        if start_index is None:
            print('Start Point could not be found in graph!')
        elif goal_index is None:
            print('Goal Point could not be found in graph')
        else:
            current = nodes[start_index[0]][start_index[1]]
            openlst.append(current)

        # Evaluate Manhattan Distance to goal
        for i in range(0, np.size(grid.cells, 0)):
            for j in range(0, np.size(grid.cells, 1)):
                nodes[i][j].calculateCostToGoal(nodes[goal_index[0]][goal_index[1]].point)

        # Search
        while openlst:

            # find item with lowest score
            current = min(openlst, key=lambda o: o.costToGo + o.costToGoal)

            # If we are at goal retrace path
            if current.point == nodes[goal_index[0]][goal_index[1]].point:
                path = []
                while current.parent:
                    path.append(current)
                    current = current.parent
                path.append(current)
                lst = []
                for i in range(len(openlst)):
                    lst.append(openlst[i].point)
                return path, lst, obstacles

            # Remove current position from open list
            openlst.remove(current)

            # Add current position to closed list
            closelst.append(current)

            # Generate children of current spot
            for node in children(current, nodes):
                if node in closelst:
                    continue
                if node in openlst:
                    # check if we beat the G score
                    current.calculateMoveCost(node.point)
                    new_cost = current.costToGo + current.costToGoal
                    if node.costToGo > new_cost:
                        node.costToGo = new_cost
                        node.parent = current
                else:

                    # Determine if point is blocked or not
                    if node.isOccupied:
                        obstacles.append(node)
                        continue
                    else:
                        # if it isn't in the open set, calculate score
                        current.calculateMoveCost(node.point)
                        node.costToGo = current.costToGo + current.costToGoal

                        # assign Parent node
                        node.parent = current

                        # Add to openlst
                        openlst.append(node)

        if len(self.path) == 0:
            raise ValueError('No Path Found')
        else:
            return [], [], []





