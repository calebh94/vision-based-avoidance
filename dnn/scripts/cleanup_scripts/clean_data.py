import pandas as pd
import numpy as np
import os

csv_file = "/home/charris/Desktop/data_collector/2019-04-06_13-50-14/training_data_obstacle_avoidance.csv"

data = pd.read_csv(csv_file)
#
# print(data)
#
# print(data.iloc[:,14:16].values)

type = 'delete'
# type = 'replace'


# for j in range(len(data.iloc[6441:-1,14:16])):
#     data.iloc[j + 6441, 14] = data.iloc[j + 6441, 1]
#     data.iloc[j + 6441, 15] = data.iloc[j + 6441, 2]




if type == 'delete':

    row = 1
    for i,x in data.iloc[:,14:16]:
        if x == '[]' or y == '[]':
            print(x)
            print(y)
            # delete row
            data.drop([row])
            print("Found empty values, dropping row {}".format(row))

        row += 1

elif type == 'replace':

    row = 1
    for i, x, y in data.iloc[:, 14:15]:
        if x == '[]' or y == '[]':
            print(x)
            print(y)
            # delete row
            # data.drop([row])
            print("Found empty values, dropping row {}".format(row))

        row += 1

data.to_csv("/home/charris/Desktop/data_collector/2019-04-06_13-50-14/training_data_obstacle_avoidance_cleaned.csv", index=False)


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
