#!/usr/bin/env python

'''

Rosbag recording for DAgger data collection

@author:  Caleb Harris <charris92@gatech.edu>


'''

import datetime
import os

import rosbag
from std_msgs.msg import Int32, String


test_name = "test"
datename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
filename = test_name + "_" + datename
bag = rosbag.Bag(filename)

# for topic, msg, t in bag.read_messages(topics='')