import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
df = pd.read_csv("elevation_test_data_1.txt", header= None)

print(df.head())
latitude = df.iloc[:,0]
longitude = df.iloc[:,1]
altitude = df.iloc[:,2]

# for i in range(len(latitude)):
#     df.iloc[i, 0] = 6378000 * np.sin((90 - (float(latitude.iloc[i]))) * np.pi / 180) * np.cos((float(latitude.iloc[i])) * np.pi / 180)
#     df.iloc[i, 1] = 6378000 * np.sin((90 - (float(latitude.iloc[i]))) * np.pi / 180) * np.sin((float(longitude.iloc[i])) * np.pi / 180)

# df = df.to_numpy()

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(projection='3d')
ax.plot3D(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c = 'r')
plt.show()
