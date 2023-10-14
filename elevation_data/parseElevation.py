import numpy
import pandas as pd
with open('elevation_test_data_1.txt', 'r') as file:

    df = pd.read_csv("elevation_test_data_1.txt", sep = '\t')
    
    latitude = df.iloc[:,1]
    longitude = df.iloc[:,2]
    altitude = df.iloc[:,3]

    length = len(latitude)
    xCoordinates = pd.Series()
    yCoordinates = pd.Series()

    for i in range(len(latitude)):
        xCoordinates[i] = 6378000 * numpy.sin((90 - (float(latitude[i]))) * numpy.pi / 180) * numpy.cos((float(latitude[i])) * numpy.pi / 180)
        yCoordinates[i] = 6378000 * numpy.sin((90 - (float(latitude[i]))) * numpy.pi / 180) * numpy.sin((float(longitude[i])) * numpy.pi / 180)
