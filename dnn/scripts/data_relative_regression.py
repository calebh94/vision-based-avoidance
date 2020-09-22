import pandas as pd
import numpy as np
import os

csv_file = "/home/charris/Desktop/dataset_4-23/2019-04-25-11-26-41.csv"
data = pd.read_csv(csv_file)


goal_x = 5
goal_y = 0

# max_delta_x = np.max(abs(goal_x - data.iloc[:,2]))
# max_delta_y = np.max(abs(goal_y - data.iloc[:,3]))

for i in range(len(data.iloc[:,15])):
    (x_comm,y_comm) = data.iloc[i,15:17]
    (x_pos, y_pos) = data.iloc[i,2:4]
    delta_x_comm = x_comm - x_pos
    delta_y_comm = y_comm - y_pos

    delta_x_norm = (goal_x - x_pos) / 10.0
    delta_y_norm = (goal_y - y_pos) / 10.0

    data.iloc[i,2] = delta_x_norm
    data.iloc[i,3] = delta_y_norm

    data.iloc[i,15] = delta_x_comm
    data.iloc[i,16] = delta_y_comm

    # if delta_x_comm < -0.1:
    #     data.drop(index = i, inplace=True)

# data_new = data[data.iloc[:,15] > -0.1]

data.to_csv("/home/charris/Desktop/dataset_4-23/dataset_4-23_v5_case2.csv", index=False)
