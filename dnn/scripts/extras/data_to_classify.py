import pandas as pd
import numpy as np
import os

csv_file = "/home/charris/Desktop/datasets/dataset_4-15_v1/dataset_4-15.csv"

data = pd.read_csv(csv_file)

max_x = np.max(data.iloc[:,2])
max_y = np.max(data.iloc[:,3])

for i in range(len(data.iloc[:,15])):
    (x_goal,y_goal) = data.iloc[i,15:17]
    (x_pos, y_pos) = data.iloc[i,2:4]
    x = x_goal - round(x_pos)
    y = y_goal - round(y_pos)

    x_norm = x_pos / max_x
    y_norm = y_pos / max_y
    data.iloc[i,27] = x_norm
    data.iloc[i,28] = y_norm

    if ((x) < 0 and (y) == 0):
        data.iloc[i,26] = 0
    elif ((x) < 0 and (y) < 0):
        data.iloc[i,26] = 1
    elif ((x) == 0 and (y) < 0):
        data.iloc[i,26] = 2
    elif ((x) > 0 and (y) < 0):
        data.iloc[i,26] = 3
    elif ((x) > 0 and (y) == 0):
        data.iloc[i,26] = 4
    elif ((x) > 0 and (y) > 0):
        data.iloc[i,26] = 5
    elif ((x) == 0 and (y) > 0):
        data.iloc[i,26] = 6
    elif ((x) < 0 and (y) > 0):
        data.iloc[i,26] = 7
    else:
        data.iloc[i,26] = 8

data.to_csv("/home/charris/Desktop/datasets/dataset_4-15_v1/dataset_4-15_modified.csv", index=False)
