import numpy as np
import pandas as pd
import geopandas as gpd


# Load in topo data from txt file
# topo = pd.read_csv('/Users/leoglonz/Desktop/camels_topo.txt', delimiter=';')

# Load in HUC02 reg2 shapefile
data = gpd.read_file('/Users/leoglonz/Desktop/Region_02_nhru_simplify_100/Region_02_nhru_simplify_100.shp')

gages_671 = gpd.read_file('/Users/leoglonz/Desktop/camels_loc/HCDN_nhru_final_671.shp')

# data.keys(), #data['hru_id'].unique()


sorted_gages = data[['GAGEID', 'Shape_Area']].sort_values(ascending=False, by='Shape_Area').reset_index()

# count the number of HRU in each gageid
data_grouped = (
    data[['GAGEID', 'Shape_Area']]
    .groupby('GAGEID')
    .agg(
        count=('GAGEID', 'size'),  # Count the number of rows per GAGEID
        total_shape_area=('Shape_Area', 'sum')  # Sum the Shape_Area values
    )
    .reset_index()  # Reset index to get a clean DataFrame
    .sort_values(by='count', ascending=False)  # Sort by count in descending order
).reset_index()


# Select GAGID 01664000 with the most subgages
gage_ex = data[data['GAGEID'] == '01664000'][['hru_id', 'GAGEID', 'hru_x', 'hru_y', 'geometry']]
gage_ex['centroid'] = gage_ex.geometry.centroid  # Calculate the centroid of each geometry (can also select .x and .y)

path = 'data/jrb.gpkg'
flowpaths = gpd.read_file(path, layer='flowpaths')
divides = gpd.read_file(path, layer='divides')
nexus = gpd.read_file(path, layer='nexus')

boundary = divides.union_all()

