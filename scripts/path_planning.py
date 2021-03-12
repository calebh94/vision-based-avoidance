#!/home/charris/obstacle_avoidance/src/dagger_pytorch_ros/venv/bin/python

'''

Path planning for obstacle avoidance using 2D occupancy map and 2D A*

@author:  Caleb Harris <charris92@gatech.edu>

'''

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Point, Twist, Transform, Quaternion
import quaternion
from std_msgs.msg import Header

import sys
import os
import csv
import matplotlib.pyplot as plt

import numpy as np

import math


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


class Cell:
    def __init__(self, x, y, occupancy):
        self.x = x
        self.y = y
        self.occupied = occupancy


class Grid:
    def __init__(self, length, width, origin, meters_per_node):
        self.length = length
        self.width = width
        self.mpn = meters_per_node
        self.origin = origin
        self.cells = [[[] for j in range(0, width)] for i in range(0, length)]

    def add_cell(self, x, y, occ):
        self.cells[x][y] = Cell(x, y, occ)


def children(point, world):
    x, y = point.point
    links = []
    neighbors = [(x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y - 1),
                 (x - 1, y + 1),
                 ]
    for d in neighbors:
        if d[0] >= np.size(world, 0) or d[1] >= np.size(world, 1):
            continue
        elif d[0] < 0 or d[1] < 0:
            continue
        if world[d[0]][d[1]] is not None:
            links.append(world[d[0]][d[1]])

    return links


def astar2d(start, goal, grid):
    # round start and goal
    start = (round(start[0], 1), round(start[1], 0))
    goal = (round(goal[0], 1), round(goal[1], 0))
    # goal = (0,1)

    # Create open and closed lists
    openlst = []
    closelst = []

    nodes = [[[] for i in range(np.size(grid.cells, 1))] for j in range(np.size(grid.cells, 0))]
    obstacles = []

    for i in range(0, np.size(grid.cells, 0)):
        for j in range(0, np.size(grid.cells, 1)):
            nodes[i][j] = Node(0, (int(grid.cells[i][j].x), int(grid.cells[i][j].y)), grid.cells[i][j].occupied)

    # Update node global values with references from grid
    for i in range(0, np.size(grid.cells, 0)):
        for j in range(0, np.size(grid.cells, 1)):
            new_x = nodes[i][j].point[0] * grid.mpn + grid.origin[0]
            new_y = nodes[i][j].point[1] * grid.mpn + grid.origin[1]
            # if nodes[i][j].isOccupied:
            #     new_x = new_x + grid.mpn/2
            #     new_y = new_y + grid.mpn/2
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

    raise ValueError('No Path Found')


def find_path(graph, start, goal, debug=False):
    # start = (0,8)
    # TODO:  Change to global waypoint, and if it doesn't exist output NO PATH
    # goal = (3,8)

    # Check if start and goal are in path

    path, list, obstacles = astar2d(start, goal, graph)
    # start = (start[0] + graph.origin[0], start[1] + graph.origin[1])
    # goal = (goal[0] + graph.origin[0], goal[1] + graph.origin[1])
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

    if debug:
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

    return nodes


def create_graph(msg):
    global map_time
    map_time = rospy.Time.now().to_sec()
    grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
    grid_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
    grid_mpc = msg.info.resolution

    columns, rows = grid.shape

    graph = Grid(rows - 1, columns - 1, grid_origin, grid_mpc)

    graph_cnt = 0
    for i in range(0, rows - 1):
        for j in range(0, columns - 1):
            # x = i * grid_mpc + grid_origin[0]
            # y = j * grid_mpc + grid_origin[1]

            if graph_cnt >= rows * columns:
                ValueError('Number of nodes is greater than grid cells!')

            occupancy = grid[j, i]

            # TODO:  Change to only if it is 0, so that unknown squares will be counted us obstacles!
            if occupancy < 20:
                occupancy = False
            else:
                occupancy = True

            graph.add_cell(i, j, occupancy)

            graph_cnt = graph_cnt + 1

    return graph


def read_path():
    waypts = []

    csv_file = 'bin/simple_path_obstacle.csv'

    if (os.path.isfile(csv_file)):
        with open(csv_file) as file:
            csv_reader = csv.reader(file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                waypts.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
                line_count += 1
    else:
        FileNotFoundError('The file could not be found :\n' + csv_file)

    return waypts


def odom_callback(odom_msg):
    global waypoint_count
    global error
    global x_des, y_des, z_des, x_truth, y_truth, z_truth
    global wpt
    global waypoints, completed_waypoints, added_waypoints
    global graph
    global map_seen, map_time
    global astar_publisher
    update = False

    x_truth = float(odom_msg.pose.pose.position.x)
    y_truth = float(odom_msg.pose.pose.position.y)
    z_truth = float(odom_msg.pose.pose.position.z)

    position = (x_truth, y_truth)

    # If a waypoint exists, check if complete
    if len(waypoints) > 0:
        x_des = waypoints[0][0]
        y_des = waypoints[0][1]
        z_des = waypoints[0][2]
        theta_des = waypoints[0][3]

        error = math.sqrt((x_truth - x_des) ** 2 + (y_truth - y_des) ** 2 + (z_truth - z_des) ** 2)

        if error < 0.1:
            print('Waypoint complete:  %f, %f, %f', (x_des, y_des, z_des))
            del waypoints[0]

    else:
        if abs(x_truth - 5) < abs(x_truth - 0):
            waypoints.insert(0, (0, 0, 1.5, -45))
            # added_waypoints = []
            map_seen = False
        else:
            waypoints.insert(0, (5, 0, 1.5, -45))
            added_waypoints = []

    # If another waypoint still exists, check if new path is required and publish waypoint message to controller
    if len(waypoints) > 0:
        # added_waypoints = []  # DELETE IN FUTURE
        if added_waypoints == [] and map_seen is True or update is True:
            end = (5.5, 0)
            path = find_path(graph, start=position, goal=end, debug=False)
            end_time = rospy.Time.now().to_sec()
            time_taken = end_time - map_time
            print('time taken:  {}'.format(time_taken))
            # added_waypoints = []
            # map_seen = False # DELETE IN FUTURE
            for i in range(0, len(path)):
                added_waypoints.insert(0, (path[i][0], path[i][1], z_des, -45))
            # added_waypoints.insert(0,waypoints)
            # waypoints.append(added_waypoints)
            waypoints = added_waypoints

        # x_des = waypoints[0][0]
        # y_des = waypoints[0][1]
        # z_des = waypoints[0][2]
        # theta_des = waypoints[0][3]
        # # Create controller message for waypoint
        # wpt = MultiDOFJointTrajectory()
        # header = Header()
        # header.stamp = rospy.Time()
        # header.frame_id = 'frame'
        # wpt.joint_names.append('base_link')
        # wpt.header = header
        # quat = quaternion.from_euler_angles(0, 0, math.radians(theta_des))
        # transforms = Transform(translation=Point(x_des, y_des, z_des), rotation=quat)
        # velocities = Twist()
        # accelerations = Twist()
        # point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accelerations], rospy.Time(0))
        # wpt.points.append(point)
        # wp_publisher.publish(wpt)
        wpt = MultiDOFJointTrajectory()
        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = 'frame'
        wpt.header = header
        for k in range(0, len(waypoints)):
            x_des = waypoints[k][0]
            y_des = waypoints[k][1]
            z_des = waypoints[k][2]
            theta_des = waypoints[k][3]
            # Create controller message for waypoint
            quat = quaternion.from_euler_angles(0, 0, math.radians(theta_des))
            transforms = Transform(translation=Point(x_des, y_des, z_des), rotation=quat)
            velocities = Twist()
            accelerations = Twist()
            point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accelerations], rospy.Time(k))
            wpt.points.append(point)
            wpt.joint_names.append('base_link')
        wp_publisher.publish(wpt)

        # Goal Point
        final = Point()
        final.x = waypoints[-1][0]
        final.y = waypoints[-1][1]
        final.z = waypoints[-1][2]

        # Next Point
        next_point = Point()
        next_point.x = waypoints[0][0]
        next_point.y = waypoints[0][1]
        next_point.z = waypoints[0][2]
        astar_publisher.publish(next_point)


def grid_callback(grid_msg):
    global map_time, graph, map_seen
    # map_time = rospy.Time.now()
    map_seen = True
    graph = create_graph(grid_msg)


def planner_main():
    # Script Parameters
    global debug
    global wp_publisher
    global goal_publisher
    global astar_publisher
    global waypoints, added_waypoints
    global map_seen
    debug = True
    map_seen = False

    # ROS node initialization
    rospy.init_node('path_planner', anonymous=True)

    waypoints = read_path()
    added_waypoints = []

    # Write Topic Strings with Namespace
    if len(sys.argv) > 1:
        namespace = sys.argv[1]
    else:
        namespace = 'firefly'  # default

    subscribe_string = '/' + str(namespace) + '/ground_truth/odometry'
    publish_string = '/' + str(namespace) + '/command/trajectory'

    # NEW PUBLISHER THAT PUBLISHES THE POINT OF THE FOUND ASTAR SOLUTION (X,Y,Z)

    pub_point_string = '/' + str(namespace) + '/astar/nextpoint'

    # Create Subscribers
    subscriber_map = rospy.Subscriber("/projected_map", OccupancyGrid, grid_callback, queue_size=1)
    subscriber_odom = rospy.Subscriber(subscribe_string, Odometry, odom_callback, queue_size=10)

    # Create Publishers
    wp_publisher = rospy.Publisher(publish_string, MultiDOFJointTrajectory, queue_size=10)
    goal_publisher = rospy.Publisher('/goal', Point, queue_size=10)

    astar_publisher = rospy.Publisher(pub_point_string, Point, queue_size=10)

    rospy.loginfo(
        "\nStarting Path Planner Node for {}! \n Current planner is a 2D AStar planner!\n ---- RUNNING ----\n".format(
            namespace))

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == '__main__':
    planner_main()

