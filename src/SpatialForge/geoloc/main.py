import geopandas as gpd
from MCMCAPSUHacks2025.core.post import plot_flowpaths, live_plot
import numpy as np
import random
import logging
from geoloc_markov_anneal import GeoLocMarkovAnneal
from shapely.geometry import Point

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


### ------------------ Data paths ------------------ ###
gages_path = '/Users/leoglonz/Desktop/Region_02_nhru_simplify_100/Region_02_nhru_simplify_100.shp'
gages_671_path = '/Users/leoglonz/Desktop/camels_loc/HCDN_nhru_final_671.shp'
region_path = '/data/jrb.gpkg'
RANDOM_SEED = 42  # For reproducibility of random operations
N_GAGES = 100
### ------------------------------------------------ ###


if __name__ == "__main__":
    # Fix randomness for reproducibility
    if RANDOM_SEED:
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

    # Load gage location data
    gages_loc = gpd.read_file(gages_path)
    gages_671_loc = gpd.read_file(gages_671_path)

    # Load flowpaths, divides, and nexus from the geopackage
    flowpaths = gpd.read_file(region_path, layer='flowpaths')
    divides = gpd.read_file(region_path, layer='divides')
    nexus = gpd.read_file(region_path, layer='nexus')


    # Combine all divides to create a single external boundary for the region.
    boundary = divides.union_all()

    print("Flowpaths, divides, and nexus loaded successfully.")


    # Generate random points within the bounding box of the region.
    bounding_box = boundary.bounds  # [xmin, ymin, xmax, ymax]
    random_points = np.random.uniform(
        low=[bounding_box[0], bounding_box[1]],
        high=[bounding_box[2], bounding_box[3]],
        size=(N_GAGES * 10, 2),  # Generate extra to ensure enough are within the region.
    )

    points = [Point(x, y) for x, y in random_points]

    # Filter points that are inside the region.
    inside_points = [point for point in points if boundary.contains(point)]

    # Keep only the desired number of gages
    points0 = inside_points[:N_GAGES]

    print(f"Selected {len(points0)} points inside the region.")

    plot_flowpaths(
        points0=gages_loc.geometry.apply(lambda geom: (geom.x, geom.y)).tolist(),  # Convert to list of tuples
        divides=divides,
        flowpaths=flowpaths,
        savepath='output/starting_gage_dist_map_2.html'
    )

    mcma_model = GeoLocMarkovAnneal(
        region_boundary=boundary,
        network=flowpaths,
        initial_points=points0.copy(), #[(point.x, point.y) for point in points0],
        n_points=N_GAGES,
        temperature=25,  # Initial temperature
        boundary_buffer=0.1,  # Buffer distance from the boundary
    )

    points_pred = mcma_model.run_simulation(n_iter=40000, cooling_rate=0.9999)

    # Save predicted points to a GeoDataFrame for visualization
    gdf_points = gpd.GeoDataFrame(geometry=points_pred, crs="EPSG:4326")
    # Example: Add an ID column
    gdf_points['id'] = range(len(gdf_points))

    # Example: Add a score column (if available)
    gdf_points['score'] = [mcma_model.objective([point]) for point in points_pred]

    gdf_points.to_file("output/points_pred_new.geojson", driver="GeoJSON")

    live_plot(
        points0,
        points_pred,
        boundary,
        flowpaths,
    )

    plot_flowpaths(
        points0=gages_loc.geometry.apply(lambda geom: (geom.x, geom.y)).tolist(),  # Convert to list of tuples
        divides=divides,
        flowpaths=flowpaths,
        savepath='output/final_gage_dist_map_2.html'
    )
