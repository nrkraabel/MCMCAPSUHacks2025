### Interactive Folium plot ###
import folium
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon, MultiPolygon
from IPython.display import clear_output
import matplotlib.pyplot as plt


def plot_points(points0, divides, flowpaths, savepath='output/starting_gage_dist_map.html'):
    # Create a gdf for the points
    gdf_points = gpd.GeoDataFrame(geometry=[Point(p) for p in points0], crs="EPSG:5070")

    # Reproject to EPSG:4326 for folium
    gdf_points = gdf_points.to_crs(epsg=4326)
    gdf_boundary = divides.to_crs(epsg=4326)
    gdf_boundary = gdf_boundary.union_all()
    gdf_rivers = flowpaths.to_crs(epsg=4326)

    # Get river segment density
    scaler = MinMaxScaler(feature_range=(1, 50))
    gdf_rivers['density'] = gdf_rivers['areasqkm'] / gdf_rivers['lengthkm']
    gdf_rivers['scaled_density'] = scaler.fit_transform(gdf_rivers[['density']])

    # Create a map centered on boundary area
    centroid = gdf_boundary.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=12)

    folium.GeoJson(
        gdf_boundary,
        style_function=lambda x: {'color': 'black', 'weight': 2, 'fillOpacity': 0.1},
    ).add_to(m)

    # folium.GeoJson(
    #     gdf_rivers,
    #     style_function=lambda x: {'color': 'blue', 'weight': 0.9, 'fillOpacity': 0.2},
    # ).add_to(m)

    for _, row in gdf_rivers.iterrows():
        folium.PolyLine(
            locations=[(point[1], point[0]) for point in row.geometry.coords],
            weight=row['scaled_density']*0.2,  # Use scaled density for line width
            color='blue',
            opacity=0.7
        ).add_to(m)
        
    for point in gdf_points.geometry:
        folium.CircleMarker(
            location=[point.y, point.x],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.5,
            popup=f"Coordinates: ({point.x:.4f}, {point.y:.4f})",        
        ).add_to(m)

    m.save(savepath)

    return m


def live_plot(points0, points_pred, boundary, flowpaths=None, figsize=(10,10)):
    clear_output(wait=True)
    plt.figure(figsize=figsize)

    # Plot the region boundary
    if isinstance(boundary, Polygon):
        plt.plot(*boundary.exterior.xy, color='black', label='Region Boundary')
    elif isinstance(boundary, MultiPolygon):
        for geom in boundary.geoms:
            plt.plot(*geom.exterior.xy, color='black', label='Region Boundary')

    if flowpaths is not None:
        for _, row in flowpaths.iterrows():
            if row.geometry.geom_type == 'LineString':
                x, y = row.geometry.xy
                plt.plot(x, y, color='blue', linewidth=0.5, label='River' if not plt.gca().lines else None)
            elif row.geometry.geom_type == 'MultiLineString':
                for line in row.geometry.geoms:
                    x, y = line.xy
                    plt.plot(x, y, color='blue', linewidth=0.5, label='River' if not plt.gca().lines else None)

    # Plot the optimized points
    plt.scatter(
        [p.x for p in points0],
        [p.y for p in points0],
        color='red',
        label='Initial',
        s=10,
    )
    plt.scatter(
        [p.x for p in points_pred],
        [p.y for p in points_pred],
        color='blue',
        alpha=0.6,
        label='Final',
        s=10,
    )

    plt.title('Locating Streamflow Gages within the Juniata River Basin')
    plt.legend(loc='upper left')
    plt.show()