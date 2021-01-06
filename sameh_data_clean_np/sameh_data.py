import numpy as np
import pandas as pd

path = "C:\\Users\\nimas\\onedrive\\desktop\\personal stuff\\programming\\rss data\\measurements_sameh\\measurements\\"

Locations = pd.read_csv( path + 'bahen7_fp.txt', header=None)
Locations = Locations.to_numpy()[:,:-1]

RSS = [pd.read_csv( path + 'RSS_' + str(i[0]) +'_'+ str(i[1]) + '_None.txt' ).to_numpy() for i in Locations ]
cords = pd.read_csv(path+ 'PDA_route.txt', header = None).to_numpy()

mac_ads = [i[0][0].split(' ')[1:] for i in RSS ]
for i in mac_ads:
    i.remove('')

set_mac_ads  =  list( set(  sum( [ i[0][0].split(' ')[1:] for i in RSS ], [] ) ))

x_cordinates = np.array([float(i[:][0][-5:-3]) for i in cords])
y_cordinates = np.array([float(i[:][0][-2:]) for i in cords])

Cordinates = np.array(zip(x_cordinates, y_cordinates))
rss  = []

for i in range(len(RSS)):
    rss.append([])
    for j in range(len(RSS[i])):
        rss[i].append(np.fromstring(RSS[i][j][0], float, sep = ' '))

    del(rss[i][0])

Radio_map_full_data = np.full(shape = [224, 30, len(set_mac_ads)], fill_value= -110)

for i in range(len(rss)):
    idx_of_macs = []
    for j in mac_ads[i]:
        idx_of_macs.append(set_mac_ads.index(j))
    
    for k in range(30):
        Radio_map_full_data[i][k][idx_of_macs] = rss[i][k]


Radio_map_mean = np.mean(Radio_map_full_data, axis = 1)
Radio_map_var = np.var(Radio_map_full_data, axis = 1)



np.save('Locations', Cordinates)
np.save('Radio_map_full', Radio_map_full_data)
np.save('Radio_map_mean', Radio_map_mean)
np.save('Radio_map_var', Radio_map_var)