import pandas as pd
from IPython.display import display
from numpy import radians, sin, cos, sqrt, arctan2,concatenate


df = pd.read_csv(r'C:\Users\Lisa.Pegram\Downloads\ABA_Opp2.csv')
df.dtypes


R=6373.0

#  Display the first 12 rows of the dataset
display(df.head(n=12))
display(df.info())


latBoston=df["LatBoston"]
LongBoston=df["LongBoston"]
LatNEMBA=df["LatNemba"]
LongNEMBA=df["LongNemba"]
LatNEMortgEx=df["LatNEMortgEx"]
LongNEMortgEx=df["LongNEMortgEx"]
latitudes = df["Lat"]
longitudes = df["Longitude"]



lon1=radians(longitudes)
lon2=radians(LongBoston)

lat1=radians(latitudes)
lat2=radians(latBoston)


dlonBost = lon2 - lon1
dlat = lat2 - lat1

a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlonBost / 2)**2
c = 2 * arctan2(sqrt(a), sqrt(1 - a))
distance = R * c

print("Boston Result:", distance*.621371)

lat3=radians(LatNEMBA)
lon3=radians(LongNEMBA)
print(lon3)
print(lat3)

dlonNemba = lon3 - lon1
dlatNemba = lat3 - lat1

print(dlonNemba)
print(dlatNemba)

i = sin(dlatNemba / 2)**2 + cos(lat1) * cos(lat3) * sin(dlonNemba / 2)**2
j = 2 * arctan2(sqrt(i), sqrt(1 - i))
distance2 = R * j


print("NEMBA Result:", distance2*.621371)
lat4=radians(LatNEMortgEx)
lon4=radians(LongNEMortgEx)

dlonMortgEx= lon4 - lon1
dlatMortgEx = lat4 - lat1

f = sin(dlatMortgEx / 2)**2 + cos(lat1) * cos(lat4) * sin(dlonMortgEx / 2)**2
g = 2 * arctan2(sqrt(f), sqrt(1 - f))
distance3 = R * g
print("NEMortgEx Result:", distance3*.621371)

#d={distance,distance2,distance}
d.to_frame(distance,distance2,distance3)
#df=concatenate([distance,distance2,distance3])
#display(distance2)
#print(df)


pathnameOut = r'C:\Users\Lisa.Pegram\Desktop\SQL'
filenameOut = "distance_output.txt"
pathOut = pathnameOut + "/" + filenameOut
fileOut = open(pathOut, 'w')
distance.to_csv(fileOut)

pathnameOut = r'C:\Users\Lisa.Pegram\Desktop\SQL'
filenameOut = "distance2_output.txt"
pathOut = pathnameOut + "/" + filenameOut
fileOut = open(pathOut, 'w')
distance2.to_csv(fileOut)

pathnameOut = r'C:\Users\Lisa.Pegram\Desktop\SQL'
filenameOut = "distance3_output.txt"
pathOut = pathnameOut + "/" + filenameOut
fileOut = open(pathOut, 'w')
distance3.to_csv(fileOut)
#fileOut.write(str(distance))

pathnameOut = r'C:\Users\Lisa.Pegram\Desktop\SQL'
filenameOut = "distance4_output.txt"
pathOut = pathnameOut + "/" + filenameOut
fileOut = open(pathOut, 'w')
df.to_csv(fileOut)
#fileOut.write(str(distance))

