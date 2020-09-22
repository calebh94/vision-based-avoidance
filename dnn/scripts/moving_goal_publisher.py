#!/usr/bin/env python

import sys

import rospy
from geometry_msgs.msg import Vector3

def goal_publisher(x_goal, y_goal, z_goal):
    pub = rospy.Publisher('goal', Vector3, queue_size=10)
    rospy.init_node('Goal_publisher')
    rate = rospy.Rate(2)
    while not rospy.is_shutdown():
        my_goal = Vector3()
        my_goal.x = float(x_goal)
        my_goal.y = float(y_goal)
        my_goal.z = float(z_goal)
        pub.publish(my_goal)
        rate.sleep()

if __name__ == '__main__':
    if len(sys.argv) > 3:
        x = sys.argv[1]
        y = sys.argv[2]
        z = sys.argv[3]
    else:
        x = 5.0
        y = 0.0
        z = 1.5
    try:
        goal_publisher(x, y, z)
    except rospy.ROSInterruptException:
        pass

