import random
import logging
import numpy as np
from shapely.geometry import Point, MultiLineString
from shapely.ops import scale
from SpatialForge.core.utils.utils import (
    precompute_distances,
    update_distances,
    objective_distance,
    objective_network,
    objective_uniqueness,
    create_spatial_index,
)


class GeoLocMarkovAnneal:
    def __init__(
        self,
        region_boundary,
        network,
        initial_points,
        n_points,
        temperature=1.0,
        boundary_buffer=10,
    ):
        """Initialize the annealing process.

        Parameters:
        - region_boundary: Shapely geometry representing the region boundary.
        - initial_points: Initial list of (x, y) coordinates for gages.
        - num_gages: Number of gages to place.
        - temperature: Initial temperature for annealing.
        """
        if not hasattr(region_boundary, 'bounds'):
            raise ValueError("region_boundary must be a valid Shapely geometry.")
        if not hasattr(network, 'geometry') or network.geometry.is_empty.any():
            raise ValueError("network must be a GeoDataFrame with non-empty geometries.")
        if len(initial_points) != n_points:
            raise ValueError("initial_points must match the number of points specified by n_points.")

        self.region_boundary = region_boundary
        self.network = network
        self.network_ls = MultiLineString(
            [geom for geom in network.geometry if geom.geom_type in ['LineString', 'MultiLineString']]
        )
        self.spatial_index = create_spatial_index(
            tuple(geom for geom in network.geometry if geom.geom_type == 'LineString')
#[geom for geom in network.geometry if geom.geom_type == 'LineString']
        )
        self.points0 = initial_points.copy()
        self.points = initial_points
        self.n_points = n_points
        self.temperature = temperature
        self.boundary_buffer = boundary_buffer

        # Precompute distances between points and river segments.
        max_distance = 1000  # Maximum acceptable distance to a river
        self.point_distances = precompute_distances(
            initial_points, #[Point(x, y) for x, y in initial_points],
            [geom for geom in network.geometry if geom.geom_type == 'LineString'],
            self.spatial_index,
            max_distance,
        )

    @staticmethod
    def flatten_network(network):
        """
        Flatten MultiLineStrings into LineStrings.
        
        Parameters:
        - network: List of Shapely geometries.
        
        Returns:
        - A list of LineString objects.
        """
        flattened = []
        for geom in network:
            if geom.geom_type == 'MultiLineString':
                flattened.extend(list(geom.geoms))  # Extract individual LineStrings
            elif geom.geom_type == 'LineString':
                flattened.append(geom)
        return flattened

    def objective(self, points):
        """Calculate the objective function based on the current points."""
        # dist = objective_distance(points, self.region_boundary, self.boundary_buffer)
        dist = objective_distance(
            points,  # Convert points to Shapely Point objects
            self.point_distances,
            self.region_boundary,
            self.boundary_buffer
        )
        network_dist = objective_network(
            points, #[Point(x, y) for x, y in points],
            MultiLineString([geom for geom in self.network.geometry])
        )
        uniqueness = objective_uniqueness(
            points,
            self.point_distances,
            [geom for geom in self.network.geometry if geom.geom_type in ['LineString', 'MultiLineString']],
            spatial_index=self.spatial_index,
        )

        loss = dist + network_dist + uniqueness
        return loss
    
    def run_iteration(self):
        """Run a single iteration of the annealing process."""
        idx = random.randint(0, self.n_points - 1)
        old_point = self.points[idx]

        bounding_box = self.region_boundary.bounds
        new_point_coords = (
            random.uniform(bounding_box[0], bounding_box[2]),
            random.uniform(bounding_box[1], bounding_box[3]),
        )
        new_point = Point(new_point_coords)  # Convert tuple to Point
        normalized_new_point = Point(new_point.x * 1e-7, new_point.y * 1e-7)

        normalized_boundary = scale(
            self.region_boundary, 
            xfact=1e-7,
            yfact=1e-7,
            origin=(self.region_boundary.centroid.x * 1e-7,
                    self.region_boundary.centroid.y * 1e-7,
            )
        )
        
        # Generate new point meeting criteria.
        while (not self.region_boundary.contains(new_point)) and (not normalized_boundary.distance(normalized_new_point) < self.boundary_buffer):
            # New candidate point within the bounding box
            new_point_coords = (
                random.uniform(bounding_box[0], bounding_box[2]),
                random.uniform(bounding_box[1], bounding_box[3]),
            )
            new_point = Point(new_point_coords)  # Convert tuple to Point
            normalized_new_point = Point(new_point.x * 1e-7, new_point.y * 1e-7)
        
        # Evaluate objective before and after move.
        current_score = self.objective(self.points)
        self.points[idx] = new_point

        # Update the distances for the moved point
        max_distance = 1000  # Maximum acceptable distance to a river
        self.point_distances = update_distances(
            Point(old_point),
            Point(new_point),
            self.point_distances,
            self.network,
            self.spatial_index,
            max_distance,
        )

        new_score = self.objective(self.points)

        # Decide to accept move
        if new_score < current_score:
            return True
        else:
            # Accept worse moves with a probability based on temperature
            diff = new_score - current_score
            accept_probability = np.exp(-diff / max(self.temperature, 0))
            if abs(random.random()) < accept_probability:
                return True
            else:
                # Revert the change
                self.points[idx] = old_point
                # Revert the distances as well
                self.point_distances = update_distances(
                    Point(new_point),
                    Point(old_point),
                    self.point_distances,
                    self.network,
                    self.spatial_index,
                    max_distance,
                )
                return False

    def update_temperature(self, cooling_rate):
        """Update the temperature for simulated annealing."""
        self.temperature *= cooling_rate

    def run_simulation(self, n_iter=10000, cooling_rate=0.99):
        """Run the simulated annealing simulation."""
        for iteration in range(n_iter):


            self.run_iteration()

            # Print progress
            if iteration % 1000 == 0:
                current_score = self.objective(self.points)
                logging.info(f"Iteration {iteration}: Score = {current_score:.2f}, Temperature = {self.temperature:.4f}")

                # print(f"Iteration {iteration}: Score = {current_score:.2f}," \
                    #   f" Temperature = {self.temperature:.4f}")

            self.update_temperature(cooling_rate)

            # live_plot(
            #     self.points0,
            #     self.points,
            #     boundary,
            #     figsize=(6,6),
            # )
        return self.points
