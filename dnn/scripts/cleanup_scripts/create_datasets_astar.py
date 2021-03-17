import pandas as pd
import numpy as np
import os

dataset_name = 'dataset_4-23'
main_dir = "/home/charris/Desktop/dataset_4-23/"

dataset_split = [0.80, 0.10, 0.10] #Training, Validation, Testing


traj_count = 0
file_count = 1

data_combined = pd.array(np.empty(1))
data = pd.array(np.empty(1))

for file in os.listdir(main_dir):
    if file.endswith('.csv'):

        file_name = os.path.join(main_dir, file)

        if file_count == 1:
            data_combined = pd.read_csv(file_name)
        else:
            data = pd.read_csv(file_name)
            # data = data[]
            data_combined = data_combined.append(data,ignore_index=True)

            print('Data successfully combined into a one dataframe')

        file_count += 1

# data_combined.TRAJECTORY = data_combined.TRAJECTORY.astype(int)

# Loop through dataframe and change trajectory counts

# current_num = 0
# found = False
#
# if int(data_combined.iloc[pnt, 0]) == 1 and found == False:
#     traj_count = traj_count + 1
#     found = True
# elif int(data_combined.iloc[pnt, 0]) == 1:
#     found = True
# else:
#     found = False
# data_combined.iloc[pnt, 0] = int(data_combined.iloc[pnt, 0]) + traj_count - 1

# for pnt in range(len(data_combined)):




goal_x = 5
goal_y = 0

data_combined = data_combined.POSE_x.astype(float)
data_combined = data_combined.POSE_y.astype(float)

max_delta_x = np.max(abs(goal_x - (data_combined.iloc[2:,2])))
max_delta_y = np.max(abs(goal_y - (data_combined.iloc[2:,3])))

for i in range(len(data_combined.iloc[:,15])):
    (x_comm,y_comm) = float(data_combined.iloc[i,15:17])
    (x_pos, y_pos) = float(data_combined.iloc[i,2:4])
    delta_x_comm = x_comm - x_pos
    delta_y_comm = y_comm - y_pos

    delta_x_norm = (goal_x - x_pos) / max_delta_x
    delta_y_norm = (goal_y - y_pos) / max_delta_y

    data_combined.iloc[i,27] = delta_x_norm
    data_combined.iloc[i,28] = delta_y_norm

    data_combined.iloc[i,15] = delta_x_comm
    data_combined.iloc[i,16] = delta_y_comm

    # if delta_x_comm < -0.1:
    #     data.drop(index = i, inplace=True)

# data_new = data_combined[data_combined.iloc[:,15] > -0.1]

data_combined.to_csv("/home/charris/Desktop/dataset_4-23/dataset_4-23_modified.csv", index=False)
