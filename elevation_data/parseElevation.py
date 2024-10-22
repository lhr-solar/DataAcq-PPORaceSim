import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# format file to be read by pandas
with open('elevation_data.txt', 'r') as fin, open('elevation_data_commas_out.txt', 'w') as fout:
    s = fin.readline()
    for line in fin:
        s = line.replace('   ', ',').replace("  ", ",").rstrip(",")
        # print(s)
        fout.write(s)

# this step is dependent on ensuring altitude is in meters in the text file
df = pd.read_csv('elevation_data_commas_out.txt')

latitude = df.iloc[:, 1]
longitude = df.iloc[:, 2]
altitude = df.iloc[:, 3]

# need to fix x and y conversions

# for i in range(len(latitude)):
#     df.iloc[i, 0] = 6378000 * np.sin((90 - (float(latitude.iloc[i]))) * np.pi / 180) * np.cos((float(latitude.iloc[i])) * np.pi / 180)
#     df.iloc[i, 1] = 6378000 * np.sin((90 - (float(latitude.iloc[i]))) * np.pi / 180) * np.sin((float(longitude.iloc[i])) * np.pi / 180)

# fig = plt.figure(figsize=(10,6))
# ax = fig.add_subplot(projection='3d')
# ax.plot3D(latitude, longitude, altitude, c = 'r')
# plt.show()
