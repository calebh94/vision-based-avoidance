# vision-based-avoidance

The dagger_pytorch_Ros environment is being developed by Caleb Harris for use in the AE8900 Special Problem.  The work in this environment is for the project:  "Obstacle Avoidance for a UAS using Reinforcement Learning".  Further details on using this information will be provided below.

## Getting Started

Follow these instructions to install the project from source.

Another good source for installing the proper setup is found here:  https://idorobotics.com/2018/11/11/3d-computer-vision-in-ros/

### Prerequisites

This package is dependent RotorS, mav_mpc_control, etc... 

and pcl (http://www.pointclouds.org/documentation/tutorials/compiling_pcl_posix.php)

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
HERE IS HOW TO INSTALL THOSE!
```

And repeat

```
AND THIS!
```

## Data Collection

### Rosbag data collection
Start by running simulation with the following mav and world:
```
roslaunch rotors_gazebo obstacle_avoidance.launch mav_name:=DJI world_name:=ob
```

Then, run the path planning expert algorithm (path_planning.py)

Then run the following command:
```
rosbag record rosout tf /DJI/command/trajectory /DJI/command/motor_speed /DJI/command/roll_pitch_yawrate_thrust /DJI/ground_truth/odometry /DJI/odometry_sensor1/odometry /DJI/vi_sensor/camera_depth/camera/image_raw /DJI/vi_sensor/camera_depth/camera/image_raw/compressed /initialpose /move_base_simple/goal
```

Afterwards, the file "read_bag.py" will be used to translate the data into a dataset.

### single image data collect

To collect data for training follow the process below.

First, launch the simulation with the desired UAV and world.  The octomap server and MPC controller will also start:
```
roslaunch rotors_gazebo obstacle_avoidance.launch mav_name:=DJI world_name:=obstacle_two
```

Second, launch the Astar solution script that is for data collection: (Currently it is not available in rotors_gazebo)
```
rosrun rotors_gazebo astar_data_collection.py DJI
```

Next, launch the data collection node:
```
rosrun rotors_gazebo data_collection.py
```

Lastly, run the waypoint publisher file for data collection:
```
rosrun rotors_gazebo waypoint_publisher_file_new 0 0 1 -45 ~/test.txt __ns:=DJI
```

## Running the basic world and controller

To run the current world model and controller run:

```
roslaunch rotors_gazebo obstacle_avoidance.launch
```

To run with joystick:

```
roslaunch rotors_gazebo obstacle_avoidance_with_joy.launch
```

To run using linear mpc controller:

```
roslaunch rotors_gazebo obstacle_avoidance_linear.launch
```

Note:  Currently, the world and launch files are moved into the rotors_gazebo folders for easy development.  In the future the launch file will be updated to access the files relative to this package.


## Running the waypoint publisher

To run the waypoint publisher python script:

```
rosrun dagger_pytorch_ros waypoint_publisher.py $CSV_FILE_LOCATION
```

The csv file is organized by having the MAV namespace on the first line, then any set of waypoints with x, y, z, yaw.

```
firefly
0, 0, 0.1, 0
0, 0, 1, 0
1, 0, 1, 0
1, 1, 1, 0
0, 0, 0.1, 0
```

## Building a map with Octomap

To build a map via the Octomap package, first start the environment with any controller.  Then, start the octomap server:

```
roslaunch octomap_server octomap_mapping.launch
```

Send the vehicle along a preset trajectory for mapping, ex. obstacle_trajectory.csv

After completing this trajectory, or checking the map is complete, save the map respectively as a binary map and probability map:

```
rosrun octomap_saver mafile.bt

octomap_saver -f mapfile.ot
```

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Caleb Harris** - *Initial work* - (caleb.harris94@gatech.edu)

## License

CURRENTLY NO LICENSE I GUESS...

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* "There is good in this world Frodo, and it is worth fighting for."  - Sam Baggins, Lord of the Rings
* etc

