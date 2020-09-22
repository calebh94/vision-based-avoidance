#!/usr/bin/env python

'''

Script for running DNN Policy in DAgger framework

'''

# Subscribe to GROUND TRUTH
# Subscribe to a Global Goal Position
# Subscribe to image data

# convert ground truth and global goal to relative position input
# convert image with transformations

# Input two into Trained DNN Model to get prediction
# ANOTHER NODE WILL CHOOSE WHICH ACTION TO TAKE

#3###################3#######################################3
# REFERENCE:  https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674



import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# sys.path.remove('/home/charris/gazebo_gym_ws/src/gym-gazebo/gym_gazebo/envs/installation/catkin_ws/devel/lib/python2.7/dist-packages')
# sys.path.insert(1,'~/obstacle_avoidance/src/dagger_pytorch_ros/venv/lib/python3.6/site-packages/cv2/cv2.cpython-36m-x86_64-linux-gnu.so')

import rospy
import roslib

from nav_msgs.msg import OccupancyGrid, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Point, Twist, Transform, Quaternion, Vector3
import quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage

import cv2
from cv_bridge import CvBridge, CvBridgeError
from skimage import io, transform

from avoidance_framework import DNN_C4F4F2_Classifier2, DNN_Classifier_small, DNN_C4F4F2_Astar_small, DNN_C4F4F2_Astar
# from torchvision import transforms
import torch.nn.functional as F
import torch
from torch.autograd import Variable
# from train_DNN_for_classify import Crop, ToTensor_Norm, Rescale, ObstacleAvoidanceDataset

import sys
import os
import csv
import matplotlib.pyplot as plt

import numpy as np

import math

roslib.load_manifest('rotors_gazebo')


def crop(output_h, output_w, image):
    h, w = image.shape[:2]
    new_h, new_w = output_h, output_w

    # top = np.random.randint(0, h - new_h)
    top = h - new_h
    # left = np.random.randint(0, w - new_w)
    left = w - new_w

    image = image[top: top + new_h,
            left: left + new_w]

    return image


def rescale(output_h, output_w, image):
    h, w = image.shape[:2]
    new_h, new_w = int(output_h), int(output_w)

    image = transform.resize(image, (new_h, new_w))

    return image


def to_tensor_norm(image):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    image = F.normalize(torch.from_numpy(image),3)

    return image

def to_tensor(image):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)

    return image


class DnnPathPolicy:

    def __init__(self, namespace):
        self.namespace = namespace

        self.type = 'regression'
        self.step = 0

        self.x_goal = 5
        self.y_goal = 0
        self.z_goal = 1.5

        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0

        self.x_round = []
        self.y_round = []

        self.x_relative = []
        self.y_relative = []

        self.image = []
        self.transformed_image = []

        self.dnn_classification = []
        self.have_classification = False

        self.x_pub = []
        self.y_pub = []
        self.z_pub = []

        self.bridge = CvBridge()

        self.got_odom = False

        # Initialize subscribers
        # self.subscribe_string_odom = '/' + str(self.namespace) + '/ground_truth/odometry'
        self.subscribe_string_odom = '/' + str(self.namespace) + '/odometry_sensor1/odometry'
        self.subscribe_string_image = '/' + str(self.namespace) + '/vi_sensor/camera_depth/camera/image_raw/compressed'
        self.pub_point_string = '/' + str(self.namespace) + '/dnn/nextpoint'
        
        self.subscriber_odom = rospy.Subscriber(self.subscribe_string_odom, Odometry, self.odom_callback, queue_size=10)
        self.subscriber_image = rospy.Subscriber(self.subscribe_string_image, CompressedImage, self.callback_image)
        self.dnn_publisher = rospy.Publisher(self.pub_point_string, Point, queue_size=10)
        self.goal_subscriber = rospy.Subscriber('/goal', Vector3, self.goal_callback, queue_size=5)

        # self.model = []
        self.init = False
        self.init_model()


    def get_path_command(self, selection):
        if selection == 0:
            x = -1
            y = 0
        elif selection == 1:
            x = -1
            y = -1
        elif selection == 2:
            x = 0
            y = -1
        elif selection == 3:
            x = 1
            y = -1
        elif selection == 4:
            x = 1
            y = 0
        elif selection == 5:
            x = 1
            y = 1
        elif selection == 6:
            x = 0
            y = 1
        elif selection == 7:
            x = -1
            y = 1
        elif selection == 8:
            x = 0.2
            y = 0
        else:
            x = 0
            y = 0

        x = self.x_pos + x
        y = self.y_pos + y

        return x,y


    def init_model(self):
        if self.type == 'classify':
            model = DNN_Classifier_small()
            device = "cuda"
            model.to(device)
            model.load_state_dict(
                torch.load('/home/charris/obstacle_avoidance/src/dagger_pytorch_ros/dnn/Models/dnn_classify_4-15_v1.pt'))
            model.eval()
        elif self.type == 'regression':
            # model = DNN_C4F4F2_Astar()
            model = DNN_C4F4F2_Astar_small()
            device = "cuda"
            model.to(device)
            model.load_state_dict(
                torch.load('/home/charris/obstacle_avoidance/src/dagger_pytorch_ros/dnn/Models/dnn_4-23_v4_30.pt'))
            model.eval()

        self.model = model
        self.init = True


    def do_transforms(self, image):
        if self.type == 'regression':
            image = crop(360, 640, image)
            image = rescale(64,64,image)
            image = image.reshape((64,64,1))
            image = to_tensor(image)
            # image = F.normalize(image,3)
        elif self.type == 'classify':
            image = crop(360, 640, image)
            image = rescale(64, 64, image)
            image = image.reshape((64, 64, 1))
            image = to_tensor(image)

        return image


    def callback_image(self, data):
        cv_image = []
        cv_image_gs = []

        try:

            np_arr = np.fromstring(data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if self.type == 'classify' or self.type == 'regression':
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            cv2.imshow('test', cv_image)
            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

        self.image = np.asarray(cv_image, dtype=np.uint8)
        self.transformed_image = self.do_transforms(self.image)

        # DO PREDICTION!
        if (self.got_odom and self.init):
            # setting up pose (x,y) input
            x_rel = self.x_goal - self.x_pos
            y_rel = self.y_goal - self.y_pos
            pose_input_norm = np.asarray([x_rel / 10.0, y_rel / 10.0])
            # norm = np.linalg.norm(pose_input_np)
            # if norm == 0:
            #     pose_input_norm = pose_input_np
            # else:
            #     pose_input_norm = pose_input_np / norm
            pose_input = Variable((torch.from_numpy(pose_input_norm)))

            # setting up image input
            img_input = Variable(self.transformed_image)


            img_input = img_input.reshape((1,img_input.shape[0],img_input.shape[1],img_input.shape[2]))
            pose_input = pose_input.reshape((1,1,pose_input.shape[0]))

            if next(self.model.parameters()).is_cuda == True:
                img_input = img_input.to(torch.device("cuda"))
                pose_input = pose_input.to(torch.device("cuda"))

            output = self.model(pose_input, img_input)
            _, pred = torch.max(output,1)

            if self.type == 'classify':

                pred_arr = pred.cpu()
                self.dnn_classification = np.asarray(pred_arr)[0]

                # self.have_classification = True
                self.x_pub, self.y_pub = self.get_path_command(self.dnn_classification)

            elif self.type == 'regression':

                action = np.asarray(output.cpu().detach().numpy())[0]
                self.x_pub = action[0] + self.x_pos
                self.y_pub = action[1] + self.y_pos

            else:
                print('No type was selected!')

            point_pub = Point()
            point_pub.x = self.x_pub
            point_pub.y = self.y_pub
            point_pub.z = self.z_goal
            self.dnn_publisher.publish(point_pub)

    def odom_callback(self, msg):

        self.x_pos = msg.pose.pose.position.x
        self.y_pos = msg.pose.pose.position.y
        self.z_pos = msg.pose.pose.position.z
        # self.pose_q1 = msg.pose.pose.orientation.x
        # self.pose_q2 = msg.pose.pose.orientation.y
        # self.pose_q3 = msg.pose.pose.orientation.z
        # self.pose_q4 = msg.pose.pose.orientation.w
        # self.xdot = msg.twist.twist.linear.x
        # self.ydot = msg.twist.twist.linear.y
        # self.zdot = msg.twist.twist.linear.z
        # self.p = msg.twist.twist.angular.x
        # self.q = msg.twist.twist.angular.y
        # self.r = msg.twist.twist.angular.z

        # self.x_round = round(self.x_pos)
        # self.y_round = round(self.y_pos)
        #
        # self.x_relative = self.x_goal - self.x_round
        # self.y_relative = self.y_goal - self.y_round

        self.got_odom = True

    def goal_callback(self, msg):

        self.x_goal = msg.x
        self.y_goal = msg.y
        self.z_goal = msg.z


def main():
    rospy.init_node('DNN_Path_Policy')

    if (len(sys.argv) > 1):
        namespace = sys.argv[1]
    else:
        namespace = 'DJI'

    DnnPathPolicy(namespace=namespace)
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
