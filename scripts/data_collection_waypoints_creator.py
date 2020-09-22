import numpy as np

xbounds = [-5.0,2.0]
ybounds = [-4.0,4.0]

xbounds2 = [3.5, 5.0]
ybounds2 = [-4.0, 4.0]

z = 1.0
yaw = -45.0

xarray = np.linspace(xbounds[0], xbounds[1], 80)
yarray = np.linspace(ybounds[0], ybounds[1], 80)

xarray2 = np.linspace(xbounds2[0], xbounds2[1], 36)
yarray2 = np.linspace(ybounds2[0], ybounds2[1], 36)

f = open("x_5_5_y_4_4_7500pts.txt", "w")
for i in xarray:
    for j in yarray:
        datastring = str(round(i,2)) + " " + str(round(j,2)) + " "  +\
                str(z) + " " + str(yaw) + "\n"
        f.write(datastring)

for i in xarray2:
    for j in yarray2:
        datastring = str(round(i, 2)) + " " + str(round(j, 2)) + " " + \
                     str(z) + " " + str(yaw) + "\n"
        f.write(datastring)

f.close()
