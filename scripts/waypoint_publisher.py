#!/home/charris/.virtualenvs/dagger_pytorch_ros/bin/python

import sys
print(sys.path)
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import rospy
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Point, Twist, Transform, Quaternion
import std_msgs.msg
from nav_msgs.msg import Odometry
import numpy as np

import quaternion

import csv
import os
import math


def readCSV():
    ns = 'firefly' #default
    waypts = np.empty([0, 4], float)

    if len(sys.argv) == 2:
        csv_file = sys.argv[1]
    elif len(sys.argv) > 2:
        csv_file = ''
        ValueError('waypoint_publisher uses a single argument, the CSV file of waypoints\n'
                   'input = ' + sys.argv)
    else:
        csv_file = '../bin/trajectory_test.csv'

    if (os.path.isfile(csv_file)):
        with open(csv_file) as file:
            csv_reader = csv.reader(file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    ns = row[0]
                    line_count+=1
                else:
                    waypts = np.append(waypts, np.array([[float(row[0]),float(row[1]),float(row[2]), float(row[3])]]), axis=0)
                    line_count+=1
    else:
        FileNotFoundError('The file could not be found :\n' + csv_file)

    return ns, waypts


def odom_callback(odom_msg):
    global waypoint_count
    global error
    global x,y,z
    global wpt

    x_truth = float(odom_msg.pose.pose.position.x)
    y_truth = float(odom_msg.pose.pose.position.y)
    z_truth = float(odom_msg.pose.pose.position.z)


    wpt = MultiDOFJointTrajectory()

    header = std_msgs.msg.Header()
    header.stamp = rospy.Time()
    header.frame_id = 'frame'
    wpt.joint_names.append('base_link')
    wpt.header = header

    if error < 0.1:
        print('Waypoint ' + str(waypoint_count) + ' complete')
        waypoint_count += 1

    if waypoint_count >= np.size(waypoints, axis=0) and error < 0.1:
        print('Waypoint list complete!\n'
              'Staying at last waypoint: ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
        waypoint_count=4

    x = waypoints[waypoint_count][0]
    y = waypoints[waypoint_count][1]
    z = waypoints[waypoint_count][2]
    theta = waypoints[waypoint_count][3]

    error = math.sqrt((x_truth-x)**2 + (y_truth-y)**2 + (z_truth-z)**2)

    quat = quaternion.from_euler_angles(0, 0, math.radians(theta))
    # quaternion = Quaternion()

    transforms = Transform(translation=Point(x,y,z), rotation=quat)

    velocities = Twist()
    accelerations = Twist()

    point = MultiDOFJointTrajectoryPoint([transforms], [velocities], [accelerations], rospy.Time(2))

    wpt.points.append(point)

    wp_publisher.publish(wpt)


def main():
    rospy.init_node("waypy_publisher")

    global waypoints
    global waypoint_count
    global error
    error = 100
    waypoint_count = 0

    namespace, waypoints = readCSV()

    subscribe_string = '/' + str(namespace) + '/ground_truth/odometry'

    odom_subscriber = rospy.Subscriber(subscribe_string, Odometry, odom_callback, queue_size=10)

    publish_string = '/' + str(namespace) + '/command/trajectory'

    global wp_publisher

    wp_publisher = rospy.Publisher(publish_string, MultiDOFJointTrajectory, queue_size=10)



    print(waypoints)

    while not rospy.is_shutdown():
        rospy.spin()


if __name__ == "__main__":
    main()
