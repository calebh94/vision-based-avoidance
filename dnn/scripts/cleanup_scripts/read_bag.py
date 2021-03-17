'''


Script to read bag files and create datasets for DNN training

Caleb Harris <charris92@gatech.edu>

'''
import os
import csv

import rosbag
from rosbag import ROSBagUnindexedException
from nav_msgs.msg import Odometry
from mav_msgs.msg import RollPitchYawrateThrust, Actuators
from geometry_msgs.msg import Point, Transform, Quaternion
from train_DNN import ObstacleAvoidanceDataset
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Point, Twist, Transform, Quaternion

import numpy as np
import cv2

image_record = True

main_dir = '/home/charris/Desktop/data_bags'

for file in os.listdir(main_dir):
    if file.endswith('.bag'):
        try:
            bag = rosbag.Bag(os.path.join(main_dir, file))
        except ROSBagUnindexedException:
            continue
        bag_name = file.replace('.bag','')

        # bag = rosbag.Bag('/media/charris/DISCOVERY/rosbags/2019-04-11-09-40-21.bag')
        # bag_name = '2019-04-11-09-40-21'
        data_dir = '/home/charris/Desktop/dataset_4-23/' + bag_name
        if os.path.exists(data_dir) is False:
            os.mkdir(data_dir)
        filename = bag_name + '.csv'

        csv_save_complete_name = os.path.join(data_dir, filename)
        with open(csv_save_complete_name, 'a') as csvfile:
            infowriter = csv.writer(csvfile)
            infowriter.writerow(("TRAJECTORY", "IMAGENAME", "POSE_x", "POSE_y", "POSE_z", "POSE_q1", "POSE_q2", "POSE_q3", "POSE_q4",
                                 "POSE_xdot", "POSE_ydot", "POSE_zdot", "POSE_p", "POSE_q", "POSE_r",
                                 "Command_x", "Command_y", "Command_z", "MPC_roll", "MPC_pitch", "MPC_yaw", "MPC_thr",
                                 "Motor_speed1", "Motor_speed2", "Motor_speed3", "Motor_speed4"))

        # READ BAG FOR IMAGES, ODOMETRY, ACTIONS
        # READ A TRAJECTORY AND STORE IT IN A BATCH OF DATA

        ### RATES ###
        # image - 20 Hz
        # ground truth - 100 Hz
        # odometry - 100 Hz
        # command_traj - 100+ Hz
        # command_motor speed - 100 Hz
        # command_rpyt - 100 Hz

        first_t = 0
        time_got_image = 1000
        ignore_data = False
        got_image = False
        got_odom = False
        got_rates = False
        got_motor = False
        got_command = False
        cnt = 1
        img_cnt = 0
        ind = 1
        arr = np.asarray([[0,0,ind]])
        for topic, msg, t in bag.read_messages(topics=['/DJI/vi_sensor/camera_depth/camera/image_raw/compressed',
                                                       '/DJI/command/trajectory',
                                                       '/DJI/ground_truth/odometry',
                                                       '/DJI/odometry_sensor1/odometry',
                                                       '/DJI/command/motor_speed',
                                                       '/DJI/command/roll_pitch_yawrate_thrust']):

            if cnt == 1:
                first_t = t.to_sec()
            cnt += 1
            print(t.to_sec() - first_t)

            if ignore_data:
                # check if back at start
                if topic == '/DJI/ground_truth/odometry':
                    x = msg.pose.pose.position.x
                    if abs(x - 0) < 0.1:
                        print('START OF TRAJECTORY')
                        ignore_data = False

            else:
                # GRAB DATA

                if topic == '/DJI/vi_sensor/camera_depth/camera/image_raw/compressed':
                    got_image = True
                    img_cnt += 1
                    np_arr = np.fromstring(msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    # cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    # cv2.imshow('test', cv_image_rgb)
                    imagename = bag_name + '_' + str(t.to_sec()) + '_' + str(ind) + '.jpg'
                    if image_record:
                        cv2.imwrite(data_dir + '/' + imagename, cv_image)
                    time_got_image = t.to_sec()

                if got_image == True:

                    if topic == '/DJI/ground_truth/odometry':
                        # arr = np.insert(arr, 1, [x,t.to_sec()-first_t,ind], axis=0)
                        pose_x = msg.pose.pose.position.x
                        pose_y = msg.pose.pose.position.y
                        pose_z = msg.pose.pose.position.z
                        pose_q1 = msg.pose.pose.orientation.x
                        pose_q2 = msg.pose.pose.orientation.y
                        pose_q3 = msg.pose.pose.orientation.z
                        pose_q4 = msg.pose.pose.orientation.w
                        xdot = msg.twist.twist.linear.x
                        ydot = msg.twist.twist.linear.y
                        zdot = msg.twist.twist.linear.z
                        p = msg.twist.twist.angular.x
                        q = msg.twist.twist.angular.y
                        r = msg.twist.twist.angular.z
                        got_odom = True

                        # Check if at the end of trajectory
                        if abs(pose_x - 5) < 0.1:
                            print('END OF TRAJECTORY')
                            ignore_data = True
                            ind += 1

                    if topic == '/DJI/command/roll_pitch_yawrate_thrust':
                        rollrate = msg.roll
                        pitchrate = msg.pitch
                        yawrate = msg.yaw_rate
                        thrust = msg.thrust.z
                        got_rates = True

                    if topic == '/DJI/command/motor_speed':
                        motor1 = msg.angular_velocities[0]
                        motor2 = msg.angular_velocities[1]
                        motor3 = msg.angular_velocities[2]
                        motor4 = msg.angular_velocities[3]
                        got_motor = True

                    if topic == '/DJI/command/trajectory':
                        comm_x = msg.points[0].transforms[0].translation.x
                        comm_y = msg.points[0].transforms[0].translation.y
                        comm_z = msg.points[0].transforms[0].translation.z
                        got_command = True

                    if got_rates and got_odom and got_motor and got_command:
                        with open(csv_save_complete_name, 'a') as csvfile:
                            infowriter = csv.writer(csvfile)
                            astar_x = []
                            astar_y = []
                            astar_z = []
                            infowriter.writerow(
                                (str(ind),
                                 str(imagename), str(pose_x), str(pose_y), str(pose_z), str(pose_q1),
                                 str(pose_q2), str(pose_q3), str(pose_q4), str(xdot),
                                 str(ydot), str(zdot), str(p), str(q), str(r),
                                 str(comm_x), str(comm_y), str(comm_z), str(rollrate),
                                     str(pitchrate), str(yawrate), str(thrust), str(motor1),
                                 str(motor2), str(motor3), str(motor4)))
                            got_image = False
                            got_motor = False
                            got_odom = False
                            got_rates = False
                            got_command = False

        print('Total Images: {}'.format(img_cnt))
        print('Total topics: {}'.format(cnt))
        print('Total time: {} s'.format(t.to_sec() - first_t))
        bag.close()
