#!/usr/bin/env python


'''

Path planning for obstacle avoidance using 2D occupancy map and 2D A*

@author:  Caleb Harris <charris92@gatech.edu>

'''

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Point, Twist, Transform, Quaternion
import quaternion
from std_msgs.msg import Header, String


import sys
import os
import csv
import matplotlib.pyplot as plt

import numpy as np

import math

from astar2d import Astar2D

# DEFINING DATA COLLECTOR CLASS
class AstarDataCollector:

    def __init__(self, namespace, waypoints):
        self.namespace = namespace

        self.detector_switch = 'off'
        self.gotOdom = False

        self.map_time = rospy.Time.now()
        self.map_seen = False

        self.graph = []
        self.x_goal = []
        self.y_goal = []
        self.z_goal = []
        self.x_pos = []
        self.y_pos = []
        self.z_pos = []

        self.added_waypoints = []
        self.waypoints = waypoints

        self.initgoal()

        self.astar = Astar2D(debug=False)

        # Initialize subscribers
        self.subscribe_string = '/' + str(self.namespace) + '/ground_truth/odometry'
        self.pub_point_string = '/' + str(self.namespace) + '/astar/nextpoint'

        self.detector_switch_sub = rospy.Subscriber("/switch_update", String, self.callback_detector_switch)

        self.subscriber_map = rospy.Subscriber("/projected_map", OccupancyGrid, self.grid_callback, queue_size=1)
        self.subscriber_odom = rospy.Subscriber(self.subscribe_string, Odometry, self.odom_callback, queue_size=10)
        self.astar_publisher = rospy.Publisher(self.pub_point_string, Point, queue_size=10)


    def initgoal(self):
        self.x_goal = 5
        self.y_goal = 0
        self.z_goal = 1.5

    def callback_detector_switch(self, msg):

        if msg.data == 'on':
            self.detector_switch = 'on'
            print("Detector Switch Off")
            # self.img_proc_status = 'collecting_data'
        else:
            self.detector_switch = 'off'
            print("Detector Switch Off")
            self.x_pos = []
            self.y_pos = []
            # self.img_proc_status = 'traveling'
            # self.reset_data()

    def grid_callback(self, grid_msg):

        self.map_time = rospy.Time.now()
        self.map_seen = True

        self.detector_switch = 'on'  # THIS IS FORCE ON FOR DAGGER TESTING
        if self.gotOdom and self.detector_switch == 'on':
            self.graph = create_graph(grid_msg)
            self.astar.run((self.x_pos, self.y_pos), (self.x_goal, self.y_goal), self.graph)
            nextpoint = Point()
            nextpoint.x = self.astar.nextpoint[0]
            nextpoint.y = self.astar.nextpoint[1]
            nextpoint.z = self.z_goal
            print("Found next point, publishing now:  " +
                  str(nextpoint.x) + " " + str(nextpoint.y))
            self.astar_publisher.publish(nextpoint)

    def odom_callback(self, odom_msg):

        x_truth = float(odom_msg.pose.pose.position.x)
        y_truth = float(odom_msg.pose.pose.position.y)
        z_truth = float(odom_msg.pose.pose.position.z)

        self.x_pos = x_truth
        self.y_pos = y_truth

        # position = (x_truth, y_truth)
        self.gotOdom = True

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
        self.cells = [[[] for j in range(0,width)] for i in range(0,length)]

    def add_cell(self, x, y, occ):
        self.cells[x][y] = Cell(x, y, occ)


def create_graph(msg):
    grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
    grid_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
    grid_mpc = msg.info.resolution

    columns, rows = grid.shape

    graph = Grid(rows-1, columns-1, grid_origin, grid_mpc)

    graph_cnt = 0
    for i in range(0, rows - 1):
        for j in range(0, columns - 1):
            # x = i * grid_mpc + grid_origin[0]
            # y = j * grid_mpc + grid_origin[1]

            if graph_cnt >= rows * columns:
                ValueError('Number of nodes is greater than grid cells!')

            occupancy = grid[j, i]

            #TODO:  Change to only if it is 0, so that unknown squares will be counted us obstacles!
            if occupancy < 20:
                occupancy = False
            else:
                occupancy = True

            graph.add_cell(i,j,occupancy)

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



def main():

    # Script Parameters
    # global debug
    # global wp_publisher
    # global goal_publisher
    # global astar_publisher
    # global waypoints, added_waypoints
    # global map_seen
    # global astar
    # global x_pos
    # global y_pos
    # global x_goal
    # global y_goal
    # global gotOdom
    # debug = True
    # map_seen = False
    #
    # # ROS node initialization
    # rospy.init_node('path_planner', anonymous=True)
    #
    # waypoints = read_path()
    # added_waypoints = []
    #
    # astar = Astar2D(debug=False)
    #
    # x_pos = []
    # y_pos = []
    # gotOdom = False
    # x_goal = 5
    # y_goal = 0
    #
    #   publish_string = '/' + str(namespace) + '/command/trajectory'
    #
    # # NEW PUBLISHER THAT PUBLISHES THE POINT OF THE FOUND ASTAR SOLUTION (X,Y,Z)
    #
    #
    # # Create Subscribers
    # subscriber_map = rospy.Subscriber("/projected_map", OccupancyGrid, grid_callback, queue_size=1)
    # subscriber_odom = rospy.Subscriber(subscribe_string, Odometry, odom_callback, queue_size=10)
    # subscriber_request = rospy.Subscriber("/")
    #
    #
    # # Create Publishers
    # wp_publisher = rospy.Publisher(publish_string, MultiDOFJointTrajectory, queue_size=10)
    # goal_publisher = rospy.Publisher('/goal', Point, queue_size=10)



    # Write Topic Strings with Namespace
    if len(sys.argv) > 1:
        namespace = sys.argv[1]
    else:
        namespace = 'DJI'

    waypoints = read_path()

    # rospy.init_node('path_planner', anonymous=True)
    # astar = Astar2D(debug=False)

    # AstarData = AstarDataCollector(namespace, waypoints)

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.init_node('Astar_Solution', anonymous=True)
        AstarDataCollector(namespace, waypoints)
        rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

