import pandas as pd
import numpy as np
import os

csv_file = "/home/charris/Desktop/data_collector/2019-04-06_13-50-14/training_data_obstacle_avoidance.csv"

data = pd.read_csv(csv_file)


row = 1
for i in range(len(data.iloc[:,0])):
    filename = str(data.iloc[i,0]).split('/')[-1]
    data.iloc[i,0] = filename
    row += 1


data.to_csv("/home/charris/Desktop/data_collector/2019-04-06_13-50-14/training_data_obstacle_avoidance_fixedfilenames.csv", index=False)


# LOOK FOR IMAGE FILE, IF IT DOESN'T EXIST THEN DELETE THAT ENTRY ********
#
# csv_file = "data/data4-4/training_data_obstacle_avoidance_cleaned.csv"
#
# data = pd.read_csv(csv_file)
#
# row = 0
# for img_name in data.iloc[:,0].values:
#     if os.path.exists(img_name) == False:
#         print(img_name)
#         data.drop([row])
#         print("Found missing img, dropping row {}".format(row))
