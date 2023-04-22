import datetime
import matplotlib.pyplot as plt
import pysolar
lat, lon = 30.267153, -97.743057  # Austin, TX
timezone = datetime.timezone(datetime.timedelta(hours=-6))  # 0800 UTC
start = datetime.datetime(2023,11,5,tzinfo=timezone)  # 1 Jan 2018

date = datetime.datetime.now(datetime.timezone.utc)
print(pysolar.solar.get_altitude(lat, lon, date))

# Calculate radiation every hour for 90 days
nhr = 24*1
dates, altitudes_deg, radiations = list(), list(), list()
for ihr in range(nhr):
    date = start + datetime.timedelta(hours=ihr)
    altitude_deg = pysolar.solar.get_altitude(lat,lon,date)
    if altitude_deg <= 0:
        radiation = 0.
    else:
        radiation = pysolar.radiation.get_radiation_direct(date,altitude_deg)
    dates.append(date)
    altitudes_deg.append(altitude_deg)
    radiations.append(radiation)

date = datetime.datetime.now(datetime.timezone.utc)
print(pysolar.solar.get_altitude(lat, lon, date))

days = [ihr/24 for ihr in range(nhr)]
fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True)
axs[0].plot(days,altitudes_deg)
axs[0].set_title('Solar altitude, degrees')
axs[1].plot(days,radiations)
axs[1].set_title('Solar radiation, W/m2')
axs[1].set_xlabel('Days since ' + start.strftime('%Y/%m/%d %H:%M UTC'))
plt.show()
