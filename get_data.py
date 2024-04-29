import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from shapely.geometry import LineString


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

if __name__ == '__main__':
    geo = pd.read_csv('./data/BJ_Taxi/roadmap.geo')
    rid_gps = []
    for _, row in geo.iterrows():
        coordinates = eval(row['coordinates'])
        road_line = LineString(coordinates=coordinates)
        center_coord = road_line.centroid
        rid_gps.append((center_coord.x, center_coord.y))

    rid_gps_dict = dict()
    for rid, rid_gps_coordinates in enumerate(rid_gps):
        rid_gps_dict[rid] = rid_gps_coordinates
    with open('./data/BJ_Taxi/rid_gps.json', 'w') as file:
        json.dump(rid_gps_dict, file)

    rid_gps = np.array(rid_gps, dtype=np.float32)
    num_roads = len(rid_gps)
    dist_geo = np.zeros((num_roads, num_roads), dtype=np.float32)
    for i in tqdm(range(num_roads)):
        dist_geo[i] = haversine(rid_gps[i, 0], rid_gps[i, 1], rid_gps[:, 0], rid_gps[:, 1])
    np.save('./data/BJ_Taxi/dist_geo.npy', dist_geo)
