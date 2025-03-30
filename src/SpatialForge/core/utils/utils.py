from rtree import index
from functools import lru_cache

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


def precompute_distances(points, network, spatial_index, max_distance):
    """
    Precompute distances from points to river segments.
    Parameters:
    - points: List of Shapely Point objects.
    - network: List of Shapely LineString objects representing the river network.
    - spatial_index: R-tree index for the river network.
    - max_distance: Maximum acceptable distance to a river.
    Returns:
    - A dictionary mapping each point to its nearest river segment and distance.
    """
    point_distances = {}
    for point in points:
        candidate_indices = list(spatial_index.intersection(point.buffer(max_distance).bounds))
        min_distance = float('inf')
        closest_river = None
        for idx in candidate_indices:
            dist = network[idx].distance(point)
            if dist < min_distance and dist <= max_distance:
                min_distance = dist
                closest_river = network[idx]
        point_distances[point] = (closest_river, min_distance)  # Use Point object as key
    return point_distances


def update_distances(moved_point, new_position, point_distances, network, spatial_index, max_distance):
    """
    Update the precomputed distances after moving a point.
    Parameters:
    - moved_point: The point that was moved (Shapely Point object).
    - new_position: The new position of the moved point (Shapely Point object).
    - point_distances: Dictionary of precomputed distances.
    - network: List of Shapely LineString objects representing the river network.
    - spatial_index: R-tree index for the river network.
    - max_distance: Maximum acceptable distance to a river.
    Returns:
    - Updated point_distances dictionary.
    """
    network = [geom for geom in network.geometry if geom.geom_type in ['LineString', 'MultiLineString']]
    candidate_indices = list(spatial_index.intersection(new_position.buffer(max_distance).bounds))
    min_distance = float('inf')
    closest_river = None
    for idx in candidate_indices:
        dist = network[idx].distance(new_position)
        if dist < min_distance and dist <= max_distance:
            min_distance = dist
            closest_river = network[idx]
    point_distances[moved_point] = (closest_river, min_distance)  # Use Point object as key
    return point_distances


# def objective_distance(points, boundary, boundary_buffer=100):
#     # Penalty for distanc between points
#     distances = pdist(points)
#     min_distance = np.min(distances) if len(distances) > 0 else 0

#     # # Penalty for points too close to the boundary
#     # boundary_penalty = 0
#     # for point in points:
#     #     dist_to_boundary = boundary.distance(Point(point))
#     #     if dist_to_boundary < boundary_buffer:
#     #         boundary_penalty += (boundary_buffer - dist_to_boundary)

#     score = -min_distance #- boundary_penalty * 5
#     return score


def objective_distance(points, point_distances, boundary, boundary_buffer=100):
    """
    Calculate the penalty for distances between points and their proximity to the boundary.
    Parameters:
    - points: List of Shapely Point objects.
    - point_distances: Dictionary mapping each point to its nearest neighbor and distance.
    - boundary: Shapely geometry representing the region boundary.
    - boundary_buffer: Buffer distance from the boundary.
    Returns:
    - Penalty score based on distances.
    """
    # Find the minimum distance using precomputed distances
    min_distance = float('inf')
    for point in points:
        _, dist = point_distances.get(point, (None, float('inf')))
        if dist < min_distance:
            min_distance = dist

    # Optional: Add penalty for points too close to the boundary
    boundary_penalty = 0
    for point in points:
        dist_to_boundary = boundary.distance(Point(point))
        if dist_to_boundary < boundary_buffer:
            boundary_penalty += (boundary_buffer - dist_to_boundary)

    score = -min_distance #- boundary_penalty * 5
    return score


@lru_cache(maxsize=None)
def create_spatial_index(rivers):
    """
    Create a spatial index for the river network.

    Parameters:
    - rivers: List of Shapely LineString objects representing river flow lines.

    Returns:
    - An R-tree index for the river network.
    """
    idx = index.Index()
    for i, river in enumerate(rivers):
        if not hasattr(river, 'bounds'):
            raise ValueError(f"Invalid geometry at index {i}: {river}")
        idx.insert(i, river.bounds)
    return idx


def assign_points_to_rivers(points, point_distances, rivers, spatial_index, max_distance):
    """
    Assign each point to the nearest river flow line if within max_distance.
    Parameters:
    - points: List of (x, y) coordinates of points.
    - point_distances: Dictionary mapping each point to its nearest river segment and distance.
    - rivers: List of Shapely LineString objects representing river flow lines.
    - spatial_index: R-tree index for the river network.
    - max_distance: Maximum distance for a point to be considered associated with a river.
    Returns:
    - A dictionary mapping each river index to the list of points associated with it.
    """
    river_assignments = {i: [] for i in range(len(rivers))}
    for point_coords in points:
        point = Point(point_coords)  # Convert tuple to Point
        # Find candidate river segments using the spatial index
        candidate_indices = list(spatial_index.intersection(point.buffer(max_distance).bounds))
        min_distance = float('inf')
        closest_river_idx = None
        for idx in candidate_indices:
            # Access point_distances using the Point object
            closest_river, dist = point_distances.get(point, (None, float('inf')))
            if dist < min_distance and dist <= max_distance:
                min_distance = dist
                closest_river_idx = idx
        if closest_river_idx is not None:
            river_assignments[closest_river_idx].append(point_coords)
    return river_assignments


def objective_uniqueness(points, point_distances, network, spatial_index):
    """
    Penalize cases where more than two points are assigned to the same river segment.
    Parameters:
    - points: List of Shapely Point objects.
    - point_distances: Dictionary mapping each point to its nearest river segment and distance.
    - network: List of Shapely LineString objects representing the river network.
    - spatial_index: R-tree index for the river network.
    Returns:
    - Total penalty for overlapping points on the same river segment.
    """
    overlap_penalty = 0
    max_distance = 1000  # Maximum acceptable distance to a river (adjust as needed)

    # Assign points to rivers using the spatial index
    river_assignments = assign_points_to_rivers(
        points, #[Point(x, y) for x, y in points],  # Convert points to Point objects
        point_distances,
        network,
        spatial_index,
        max_distance
    )

    # Penalize rivers with more than two points
    for river_points in river_assignments.values():
        if len(river_points) > 2:
            overlap_penalty += (len(river_points) - 2) * 1000  # Penalty per extra point

    return overlap_penalty


def objective_network(points, network):
    """Enforce streamflow gages on river locations"""
    dist = network.distance(points)
    return sum(dist)
