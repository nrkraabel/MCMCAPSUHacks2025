import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import random
import math
from scipy.spatial import Voronoi
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import concurrent.futures
import numba
from numba import jit, prange, cuda
import os

class GerrymanderSimulator:
    def __init__(self, state_map, num_districts=13, use_gpu=False):
        """
        Initialize the simulator with a state map and number of districts.
        
        Parameters:
        - state_map: A numpy array with shape (height, width, 3) where each pixel has
                    [population, red_votes, blue_votes]
        - num_districts: Number of districts to create
        - use_gpu: Whether to use GPU acceleration (if available)
        """
        self.state_map = state_map
        self.height, self.width, _ = state_map.shape
        self.num_districts = num_districts
        self.use_gpu = use_gpu and cuda.is_available()
        self.num_cpus = max(1, os.cpu_count() - 4)  # Reserve just 4 cores for system
        print(f"Using {self.num_cpus} CPU cores for parallelization")
    
        if self.use_gpu:
            print("Using GPU acceleration")
            # Convert arrays to CUDA-compatible format if using GPU
            self.state_map_gpu = cuda.to_device(self.state_map)
        # else:
        #     # print("Using CPU processing")

        
        # Create a mask of valid (non-zero population) pixels
        self.valid_mask = (state_map[:,:,0] > 0)
        
        # Initialize the district map randomly using Voronoi tessellation
        self.district_map = np.zeros((self.height, self.width), dtype=np.int32)
        self.initialize_districts()
        
        # Pre-compute the neighbor map for faster lookup during simulation
        self._precompute_neighbor_map()
        
        # Keep track of district statistics
        self.district_stats = {
            'population': np.zeros(num_districts),
            'red_votes': np.zeros(num_districts),
            'blue_votes': np.zeros(num_districts),
            'center_x': np.zeros(num_districts),
            'center_y': np.zeros(num_districts),
            'perimeter': np.zeros(num_districts),
            'area': np.zeros(num_districts)
        }
        
        # Calculate initial stats
        self.calculate_all_district_stats()
        
        # Parameters for the algorithm
        self.temperature = 100000
        self.cooling_rate = 0.9999
        self.phase = 1
        
        # Metric weights (will be adjusted during phases)
        self.weights = {
            'population_equality': 10000,
            'compactness': 4,
            'center_distance': 1,
            'election_results': 5
        }
        
        # Target election results (to be set by user)
        self.target_vote_margins = None
        
        # Create a pool of workers for parallelization
        self.pool = None  # Will initialize when needed
    
    def _precompute_neighbor_map(self):
        """Precompute the neighbor map for faster lookup"""
        # print("Precomputing neighbor map...")
        self.neighbor_map = {}
        
        # For each valid cell, store its neighbors
        for i in range(self.height):
            for j in range(self.width):
                if self.valid_mask[i, j]:
                    self.neighbor_map[(i, j)] = []
                    for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                        if 0 <= ni < self.height and 0 <= nj < self.width and self.valid_mask[ni, nj]:
                            self.neighbor_map[(i, j)].append((ni, nj))
    
    def initialize_districts(self):
        """Initialize district map using Voronoi tessellation"""
        # Generate random seed points
        valid_indices = np.argwhere(self.valid_mask)
        seed_indices = valid_indices[np.random.choice(len(valid_indices), self.num_districts, replace=False)]
        
        # Create a Voronoi diagram
        vor = Voronoi(seed_indices)
        
        # Parallelize the assignment of pixels to districts
        @jit(nopython=True, parallel=True)
        def assign_pixels(height, width, valid_mask, district_map, seed_indices):
            for i in prange(height):
                for j in range(width):
                    if valid_mask[i, j]:
                        # Find the closest seed point
                        min_dist = float('inf')
                        closest_idx = 0
                        
                        for idx, seed in enumerate(seed_indices):
                            dist = (i - seed[0])**2 + (j - seed[1])**2
                            if dist < min_dist:
                                min_dist = dist
                                closest_idx = idx
                        
                        district_map[i, j] = closest_idx
            return district_map
        
        self.district_map = assign_pixels(self.height, self.width, self.valid_mask, 
                                          self.district_map, seed_indices)
        self._fill_holes()
    def _fill_holes(self):
        """
        Fill in small holes in the district map.
        A hole is defined as a zero-population area completely surrounded by districts.
        """
        # Create a mask of areas that have been assigned to districts
        district_assigned = (self.district_map >= 0) & self.valid_mask
        
        # Identify potential holes (zero-population areas)
        potential_holes = (self.state_map[:,:,0] == 0) & ~district_assigned
        
        # If there are no potential holes, return early
        if not np.any(potential_holes):
            return
        
        # Find connected components in the potential holes
        from scipy import ndimage
        labeled_holes, num_holes = ndimage.label(potential_holes)
        
        print(f"Found {num_holes} potential holes or unassigned regions")
        
        # Process each connected component
        for hole_id in range(1, num_holes + 1):
            hole_mask = (labeled_holes == hole_id)
            hole_size = np.sum(hole_mask)
            
            # Only process small holes (adjust threshold as needed)
            if hole_size > 100:  # Skip large "holes" which are likely outside state boundaries
                continue
                
            # Check if this component is surrounded by districts (a true hole)
            # Dilate the hole mask to find its neighbors
            dilated = ndimage.binary_dilation(hole_mask)
            neighbors_mask = dilated & ~hole_mask
            
            # Count how many surrounding pixels are assigned to districts
            surrounding_districts = self.district_map[neighbors_mask]
            valid_neighbors = np.sum(self.valid_mask[neighbors_mask])
            
            # If most of the surrounding pixels are assigned to districts, this is a hole
            if valid_neighbors > 0 and valid_neighbors / np.sum(neighbors_mask) > 0.5:
                print(f"Filling hole with size {hole_size}")
                
                # Get list of surrounding districts
                neighbor_districts = []
                for i, j in np.argwhere(hole_mask):
                    for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                        if (0 <= ni < self.height and 0 <= nj < self.width and 
                            self.valid_mask[ni, nj]):
                            neighbor_districts.append(self.district_map[ni, nj])
                
                # Find most common neighboring district
                if neighbor_districts:
                    from collections import Counter
                    most_common_district = Counter(neighbor_districts).most_common(1)[0][0]
                    
                    # Fill the hole with this district
                    self.district_map[hole_mask] = most_common_district
                    self.valid_mask[hole_mask] = True
        
    @staticmethod
    def calc_perimeter_chunk(params):
        """
        Calculate perimeter for a chunk of pixels
        
        Parameters:
        - params: A tuple containing (pixel_chunk, district_map, district_id, height, width)
        
        Returns:
        - The local perimeter count
        """
        pixel_chunk, district_map, district_id, height, width = params
        local_perimeter = 0
        for i, j in pixel_chunk:
            for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if 0 <= ni < height and 0 <= nj < width:
                    if district_map[ni, nj] != district_id:
                        local_perimeter += 1
        return local_perimeter
    
    def calculate_all_district_stats(self):
        """Calculate all statistics for all districts"""
        # Reset stats
        for key in self.district_stats:
            self.district_stats[key] = np.zeros(self.num_districts)
        
        # Use numpy operations for faster calculation
        district_ids = np.unique(self.district_map[self.valid_mask])
        
        for district_id in district_ids:
            mask = (self.district_map == district_id)
            
            # Count population and votes using vectorized operations
            self.district_stats['population'][district_id] = np.sum(self.state_map[:,:,0] * mask)
            self.district_stats['red_votes'][district_id] = np.sum(self.state_map[:,:,1] * mask)
            self.district_stats['blue_votes'][district_id] = np.sum(self.state_map[:,:,2] * mask)
            
            # Calculate center of population
            if self.district_stats['population'][district_id] > 0:
                # Get indices where the mask is True
                pop_indices = np.argwhere(mask & (self.state_map[:,:,0] > 0))
                pop_weights = np.array([self.state_map[i, j, 0] for i, j in pop_indices])
                
                if len(pop_indices) > 0:
                    self.district_stats['center_y'][district_id] = np.average(pop_indices[:, 0], weights=pop_weights)
                    self.district_stats['center_x'][district_id] = np.average(pop_indices[:, 1], weights=pop_weights)
            
            # Calculate perimeter more efficiently
            perimeter = 0
            district_pixels = np.argwhere(mask)
            
            # Use multiple processes to calculate perimeter if there are many pixels
            if len(district_pixels) > 10000 and self.num_cpus > 1:
                chunk_size = max(1, len(district_pixels) // self.num_cpus)
                pixel_chunks = [district_pixels[i:i+chunk_size] for i in range(0, len(district_pixels), chunk_size)]
                
                # Create parameter tuples for the static method
                params = [(chunk, self.district_map, district_id, self.height, self.width) 
                          for chunk in pixel_chunks]
                
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cpus) as executor:
                    results = list(executor.map(GerrymanderSimulator.calc_perimeter_chunk, params))
                
                perimeter = sum(results)
            else:
                # Call the static method directly with a single chunk
                params = (district_pixels, self.district_map, district_id, self.height, self.width)
                perimeter = GerrymanderSimulator.calc_perimeter_chunk(params)
            
            self.district_stats['perimeter'][district_id] = perimeter
            self.district_stats['area'][district_id] = np.sum(mask)
    
    def update_district_stats(self, pixel_i, pixel_j, old_district, new_district):
        """Update district stats when a pixel changes from old_district to new_district"""
        # Get the data for the pixel
        pixel_pop = self.state_map[pixel_i, pixel_j, 0]
        pixel_red = self.state_map[pixel_i, pixel_j, 1]
        pixel_blue = self.state_map[pixel_i, pixel_j, 2]
        
        # Update population and votes
        self.district_stats['population'][old_district] -= pixel_pop
        self.district_stats['population'][new_district] += pixel_pop
        
        self.district_stats['red_votes'][old_district] -= pixel_red
        self.district_stats['red_votes'][new_district] += pixel_red
        
        self.district_stats['blue_votes'][old_district] -= pixel_blue
        self.district_stats['blue_votes'][new_district] += pixel_blue
        
        # Update centers using the formula for adding/removing from weighted average
        if self.district_stats['population'][old_district] > 0:
            old_total_pop = self.district_stats['population'][old_district] + pixel_pop
            old_center_x = self.district_stats['center_x'][old_district]
            old_center_y = self.district_stats['center_y'][old_district]
            
            # Remove the pixel from old district center calculation
            self.district_stats['center_x'][old_district] = (old_center_x * old_total_pop - pixel_j * pixel_pop) / self.district_stats['population'][old_district]
            self.district_stats['center_y'][old_district] = (old_center_y * old_total_pop - pixel_i * pixel_pop) / self.district_stats['population'][old_district]
        
        # Add to new district center calculation
        new_total_pop = self.district_stats['population'][new_district]
        if new_total_pop > 0:
            old_new_pop = new_total_pop - pixel_pop
            if old_new_pop > 0:
                old_center_x = self.district_stats['center_x'][new_district]
                old_center_y = self.district_stats['center_y'][new_district]
                
                self.district_stats['center_x'][new_district] = (old_center_x * old_new_pop + pixel_j * pixel_pop) / new_total_pop
                self.district_stats['center_y'][new_district] = (old_center_y * old_new_pop + pixel_i * pixel_pop) / new_total_pop
            else:
                self.district_stats['center_x'][new_district] = pixel_j
                self.district_stats['center_y'][new_district] = pixel_i
        
        # Update area
        self.district_stats['area'][old_district] -= 1
        self.district_stats['area'][new_district] += 1
        
        # Update perimeter (more accurate approach)
        self._update_perimeter(pixel_i, pixel_j, old_district, new_district)
    
    def _update_perimeter(self, pixel_i, pixel_j, old_district, new_district):
        """Update perimeter stats when a pixel changes district"""
        # Get the neighbors of this pixel
        neighbors = self.neighbor_map.get((pixel_i, pixel_j), [])
        
        # Count perimeter changes for old district
        old_perimeter_change = 0
        for ni, nj in neighbors:
            if self.district_map[ni, nj] == old_district:
                # This was not a perimeter before, but now is
                old_perimeter_change += 1
            elif self.district_map[ni, nj] != new_district:
                # This was a perimeter before, but now isn't
                old_perimeter_change -= 1
        
        # Count perimeter changes for new district
        new_perimeter_change = 0
        for ni, nj in neighbors:
            if self.district_map[ni, nj] == new_district:
                # This was a perimeter before, but now isn't
                new_perimeter_change -= 1
            elif self.district_map[ni, nj] != old_district:
                # This was not a perimeter before, but now is
                new_perimeter_change += 1
        
        # Update perimeter stats
        self.district_stats['perimeter'][old_district] += old_perimeter_change
        self.district_stats['perimeter'][new_district] += new_perimeter_change
    
    @staticmethod
    @jit(nopython=True)
    def _will_break_district_numba(district_map, pixel_i, pixel_j, district_id, height, width):
        """Numba-optimized implementation of district connectivity check"""
        # Get neighbors of the same district
        neighbors = []
        for ni, nj in [(pixel_i+1, pixel_j), (pixel_i-1, pixel_j), (pixel_i, pixel_j+1), (pixel_i, pixel_j-1)]:
            if 0 <= ni < height and 0 <= nj < width and district_map[ni, nj] == district_id:
                neighbors.append((ni, nj))
        
        # If 0 or 1 neighbors, removing won't disconnect anything
        if len(neighbors) <= 1:
            return False
        
        # Pick first neighbor and try to reach others
        if len(neighbors) > 1:
            start = neighbors[0]
            
            # Use BFS to check connectivity
            visited = np.zeros((height, width), dtype=np.bool_)
            queue = [start]
            visited[start[0], start[1]] = True
            
            while queue:
                current = queue.pop(0)
                # Check neighbors
                for ni, nj in [(current[0]+1, current[1]), (current[0]-1, current[1]), 
                               (current[0], current[1]+1), (current[0], current[1]-1)]:
                    if 0 <= ni < height and 0 <= nj < width and not visited[ni, nj] and district_map[ni, nj] == district_id:
                        if ni == pixel_i and nj == pixel_j:
                            continue  # Skip the pixel we're removing
                        visited[ni, nj] = True
                        queue.append((ni, nj))
            
            # Check if all neighbors were reached
            for neighbor in neighbors[1:]:
                if not visited[neighbor[0], neighbor[1]]:
                    return True  # District would be broken
        
        return False
    
    def will_break_district(self, pixel_i, pixel_j, district_id):
        """Check if removing this pixel would break the district into disconnected parts"""
        return self._will_break_district_numba(self.district_map, pixel_i, pixel_j, district_id, 
                                               self.height, self.width)
    
    def get_boundary_pixels(self):
        """Get all pixels that are on the boundary between districts"""
        # Use vectorized operations to find pixels with different neighbors
        boundary_pixels = []
        
        # This is a hotspot for performance optimization
        # Pre-allocate arrays for the four neighbor directions
        up_shifted = np.pad(self.district_map[:-1, :], ((1, 0), (0, 0)), mode='constant', constant_values=-1)
        down_shifted = np.pad(self.district_map[1:, :], ((0, 1), (0, 0)), mode='constant', constant_values=-1)
        left_shifted = np.pad(self.district_map[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=-1)
        right_shifted = np.pad(self.district_map[:, 1:], ((0, 0), (0, 1)), mode='constant', constant_values=-1)
        
        # A pixel is on a boundary if any of its neighbors are in a different district
        is_boundary = ((up_shifted != self.district_map) | 
                      (down_shifted != self.district_map) | 
                      (left_shifted != self.district_map) | 
                      (right_shifted != self.district_map)) & self.valid_mask
        
        # Get coordinates of boundary pixels
        boundary_pixels = np.argwhere(is_boundary)
        
        return boundary_pixels
    
    def score_map(self):
        """Score the current map based on our metrics"""
        score = 0
        
        # Population equality score - use numpy for vectorized operations
        pop_std = np.std(self.district_stats['population'])
        pop_mean = np.mean(self.district_stats['population'])
        pop_score = (pop_std / pop_mean) ** 4  
        score += self.weights['population_equality'] * pop_score
        epsilon = 1e-10  # Small number to prevent division by zero
        # Compactness score (perimeter to area ratio)
        compactness_scores = self.district_stats['perimeter'] / np.sqrt(self.district_stats['area'] +epsilon )
        compactness_score = np.mean(compactness_scores)
        score += self.weights['compactness'] * compactness_score
        
        # Center distance score
        if self.weights['center_distance'] > 0:
            center_dist_score = 0
            for district_id in range(self.num_districts):
                # Use vectorized operations where possible
                mask = (self.district_map == district_id)
                if np.sum(mask) == 0:
                    continue
                    
                center_x = self.district_stats['center_x'][district_id]
                center_y = self.district_stats['center_y'][district_id]
                
                # Create meshgrid for vectorized distance calculation
                y_indices, x_indices = np.mgrid[0:self.height, 0:self.width]
                dist_grid = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
                
                # Apply mask and weights
                weighted_dist = dist_grid * self.state_map[:,:,0] * mask
                total_pop = np.sum(self.state_map[:,:,0] * mask)
                
                if total_pop > 0:
                    center_dist_score += np.sum(weighted_dist) / total_pop
            
            center_dist_score /= self.num_districts
            score += self.weights['center_distance'] * center_dist_score
        
        # Election results score
        if self.weights['election_results'] > 0 and self.target_vote_margins is not None:
            vote_margins = []
            for district_id in range(self.num_districts):
                red = self.district_stats['red_votes'][district_id]
                blue = self.district_stats['blue_votes'][district_id]
                total = red + blue
                
                if total > 0:
                    margin = red / total
                else:
                    margin = 0.5
                
                vote_margins.append(margin)
            
            # Sort margins and compare to target
            vote_margins.sort()
            target_margins = np.array(self.target_vote_margins)
            
            # Calculate mean squared error between actual and target
            margins_error = np.mean((np.array(vote_margins) - target_margins) ** 2)
            score += self.weights['election_results'] * margins_error
        
        return score
    
    def set_target_vote_distribution(self, distribution_type, red_proportion=None):
        """
        Set the target vote distribution
        
        Parameters:
        - distribution_type: 'fair', 'red_gerrymander', 'blue_gerrymander', or 'incumbent'
        - red_proportion: Overall proportion of red votes (used for fair distribution)
        """
        if red_proportion is None:
            # Calculate from the current map
            total_red = np.sum(self.state_map[:,:,1])
            total_blue = np.sum(self.state_map[:,:,2])
            red_proportion = total_red / (total_red + total_blue)
        
        margins = []
        
        if distribution_type == 'fair':
            # Create a smooth curve that crosses 50% at the popular vote percentage
            for i in range(self.num_districts):
                # Scale from 0 to 1
                x = i / (self.num_districts - 1)
                # Adjust to make sure it crosses 50% at the right place
                margin = x * 0.5 + red_proportion - 0.25
                margins.append(min(max(margin, 0.1), 0.9))  # Bound between 10% and 90%
        
        elif distribution_type == 'red_gerrymander':
            # Cluster blue votes into a few dense districts
            for i in range(self.num_districts):
                if i < self.num_districts // 4:  # Number of "packed" blue districts
                    margins.append(0.2)  # Very blue district
                else:
                    margins.append(0.55)  # Slightly red district
        
        elif distribution_type == 'blue_gerrymander':
            # Cluster red votes into a few dense districts
            for i in range(self.num_districts):
                if i < self.num_districts // 3:  # Number of "packed" red districts
                    margins.append(0.8)  # Very red district
                else:
                    margins.append(0.45)  # Slightly blue district
        
        elif distribution_type == 'incumbent':
            # No swing districts - all are safe
            for i in range(self.num_districts):
                # Scale from 0 to 1
                x = i / (self.num_districts - 1)
                # Create polarized districts
                if x < red_proportion:
                    margins.append(0.65)  # Safe red
                else:
                    margins.append(0.35)  # Safe blue
        
        self.target_vote_margins = sorted(margins)
    
    def run_batch_parallel(self, batch_size=1000, num_batches=10):
        """Run multiple batches in parallel using multiprocessing"""
        # Use all available cores with a small reserve
        if self.num_cpus is None or self.num_cpus <= 0:
            self.num_cpus = max(1, os.cpu_count() - 4)  # Reserve 4 cores for system

        # Initialize the pool if it doesn't exist
        if self.pool is None and self.num_cpus > 1:
            self.pool = mp.Pool(processes=self.num_cpus)
            print(f"Created process pool with {self.num_cpus} workers")
        
        if self.pool:
            # Prepare arguments for each worker - don't pass self, pass necessary data
            district_maps = []
            for _ in range(self.num_cpus):
                district_maps.append(self.district_map.copy())
            
            # Create sub-batch sizes to distribute work evenly
            sub_batch_size = max(10, batch_size // self.num_cpus)
            
            # Create a copy of necessary data
            state_map_copy = self.state_map.copy()
            neighbor_map_copy = {}
            for key, value in self.neighbor_map.items():
                neighbor_map_copy[key] = value.copy()
                
            valid_mask_copy = self.valid_mask.copy()
            weights_copy = self.weights.copy()
            target_margins_copy = None if self.target_vote_margins is None else self.target_vote_margins.copy()
            
            # Prepare serializable arguments for each worker
            args_list = []
            for worker_id in range(self.num_cpus):
                worker_batch_size = sub_batch_size
                if worker_id == self.num_cpus - 1:
                    worker_batch_size = batch_size - (self.num_cpus - 1) * sub_batch_size
                    
                if worker_batch_size <= 0:
                    continue
                    
                # Pass all necessary data explicitly, not 'self'
                args_list.append((
                    district_maps[worker_id],
                    worker_batch_size,
                    self.temperature,
                    worker_id,
                    state_map_copy,
                    neighbor_map_copy,
                    valid_mask_copy,
                    {k: v.copy() for k, v in self.district_stats.items()}, # Deep copy of stats
                    weights_copy,
                    target_margins_copy,
                    self.height,
                    self.width,
                    self.num_districts
                ))
            
            # Use a static method for processing to avoid passing self
            results = self.pool.map(GerrymanderSimulator._static_process_batch, args_list)
            
            # Find the best result
            best_score = self.score_map()  # Current score
            best_map = None
            best_stats = None
            total_accepted = 0
            
            for district_map, stats, accepted, score, worker_id in results:
                total_accepted += accepted
                if score < best_score:
                    best_score = score
                    best_map = district_map
                    best_stats = stats
            
            # Update the map if we found a better one
            if best_map is not None:
                self.district_map = best_map
                self.district_stats = best_stats
            
            return total_accepted
        else:
            # Fall back to single-threaded processing
            return self._process_batch(batch_size)

    @staticmethod
    def _static_process_batch(args):
        """Static worker function for parallel processing that can be pickled"""
        (district_map, batch_size, temperature, worker_id, 
        state_map, neighbor_map, valid_mask, district_stats, 
        weights, target_margins, height, width, num_districts) = args
        
        # Make a local copy
        local_district_map = district_map.copy()
        local_district_stats = {k: v.copy() for k, v in district_stats.items()}
        accepted_count = 0
        
        # Process each iteration
        for _ in range(batch_size):
            # Get boundary pixels
            boundary_pixels = GerrymanderSimulator._static_get_boundary_pixels(
                local_district_map, valid_mask, height, width)
            
            if len(boundary_pixels) == 0:
                continue
            
            # Randomly select a boundary pixel
            idx = np.random.randint(0, len(boundary_pixels))
            pixel_i, pixel_j = boundary_pixels[idx]
            old_district = local_district_map[pixel_i, pixel_j]
            
            # Find a neighboring district
            neighboring_districts = set()
            for ni, nj in neighbor_map.get((pixel_i, pixel_j), []):
                neighboring_districts.add(local_district_map[ni, nj])
            
            neighboring_districts.discard(old_district)
            
            if not neighboring_districts:
                continue
                
            new_district = random.choice(list(neighboring_districts))
            
            # Check if this would break the district
            if GerrymanderSimulator._static_will_break_district(
                    local_district_map, pixel_i, pixel_j, old_district, height, width):
                continue
            
            # Score the current map
            current_score = GerrymanderSimulator._static_score_map(
                local_district_stats, weights, target_margins, height, width, num_districts)
            
            # Make the change and score again
            local_district_map[pixel_i, pixel_j] = new_district
            GerrymanderSimulator._static_update_stats(
                local_district_stats, pixel_i, pixel_j, old_district, new_district, 
                state_map, neighbor_map, height, width)
            
            new_score = GerrymanderSimulator._static_score_map(
                local_district_stats, weights, target_margins, height, width, num_districts)
            
            # Decide whether to accept the change
            if new_score <= current_score:
                # Accept the change (it improved the score)
                accepted_count += 1
            else:
                # Decide probabilistically whether to accept a worse score
                accept_probability = np.exp(-(new_score - current_score) / temperature)
                
                if random.random() < accept_probability:
                    # Accept the change despite being worse
                    accepted_count += 1
                else:
                    # Revert the change
                    local_district_map[pixel_i, pixel_j] = old_district
                    GerrymanderSimulator._static_update_stats(
                        local_district_stats, pixel_i, pixel_j, new_district, old_district, 
                        state_map, neighbor_map, height, width)
        
        # Calculate final score
        final_score = GerrymanderSimulator._static_score_map(
            local_district_stats, weights, target_margins, height, width, num_districts)
        
        return local_district_map, local_district_stats, accepted_count, final_score, worker_id

    @staticmethod
    def _static_get_boundary_pixels(district_map, valid_mask, height, width):
        """Static method to get boundary pixels for a specific district map"""
        # Pre-allocate arrays for the four neighbor directions
        up_shifted = np.pad(district_map[:-1, :], ((1, 0), (0, 0)), mode='constant', constant_values=-1)
        down_shifted = np.pad(district_map[1:, :], ((0, 1), (0, 0)), mode='constant', constant_values=-1)
        left_shifted = np.pad(district_map[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=-1)
        right_shifted = np.pad(district_map[:, 1:], ((0, 0), (0, 1)), mode='constant', constant_values=-1)
        
        # A pixel is on a boundary if any of its neighbors are in a different district
        is_boundary = ((up_shifted != district_map) | 
                    (down_shifted != district_map) | 
                    (left_shifted != district_map) | 
                    (right_shifted != district_map)) & valid_mask
        
        # Get coordinates of boundary pixels
        boundary_pixels = np.argwhere(is_boundary)
        
        return boundary_pixels

    @staticmethod
    def _static_will_break_district(district_map, pixel_i, pixel_j, district_id, height, width):
        """Static method to check if removing this pixel would break the district"""
        # Get neighbors of the same district
        neighbors = []
        for ni, nj in [(pixel_i+1, pixel_j), (pixel_i-1, pixel_j), (pixel_i, pixel_j+1), (pixel_i, pixel_j-1)]:
            if 0 <= ni < height and 0 <= nj < width and district_map[ni, nj] == district_id:
                neighbors.append((ni, nj))
        
        # If 0 or 1 neighbors, removing won't disconnect anything
        if len(neighbors) <= 1:
            return False
        
        # Pick first neighbor and try to reach others
        if len(neighbors) > 1:
            start = neighbors[0]
            
            # Use BFS to check connectivity
            visited = np.zeros((height, width), dtype=np.bool_)
            queue = [start]
            visited[start[0], start[1]] = True
            
            while queue:
                current = queue.pop(0)
                # Check neighbors
                for ni, nj in [(current[0]+1, current[1]), (current[0]-1, current[1]), 
                            (current[0], current[1]+1), (current[0], current[1]-1)]:
                    if 0 <= ni < height and 0 <= nj < width and not visited[ni, nj] and district_map[ni, nj] == district_id:
                        if ni == pixel_i and nj == pixel_j:
                            continue  # Skip the pixel we're removing
                        visited[ni, nj] = True
                        queue.append((ni, nj))
            
            # Check if all neighbors were reached
            for neighbor in neighbors[1:]:
                if not visited[neighbor[0], neighbor[1]]:
                    return True  # District would be broken
        
        return False

    @staticmethod
    def _static_score_map(district_stats, weights, target_margins, height, width, num_districts):
        """Static method to score the map based on district stats"""
        score = 0
        
        # Population equality score
        pop_std = np.std(district_stats['population'])
        pop_mean = np.mean(district_stats['population'])
        if pop_mean > 0:
            pop_score = (pop_std / pop_mean) ** 4
            score += weights['population_equality'] * pop_score
        epsilon = 1e-10
        # Compactness score
        compactness_scores = district_stats['perimeter'] / np.sqrt(district_stats['area']+ epsilon) 
        compactness_score = np.mean(compactness_scores)
        score += weights['compactness'] * compactness_score
        
        # Center distance score (simplified for static version)
        if weights['center_distance'] > 0:
            center_dist_score = np.mean(
                np.sqrt((district_stats['center_x'] - width/2)**2 + 
                        (district_stats['center_y'] - height/2)**2)
            )
            score += weights['center_distance'] * center_dist_score * 0.01
        
        # Election results score
        if weights['election_results'] > 0 and target_margins is not None:
            vote_margins = []
            for district_id in range(num_districts):
                red = district_stats['red_votes'][district_id]
                blue = district_stats['blue_votes'][district_id]
                total = red + blue
                
                if total > 0:
                    margin = red / total
                else:
                    margin = 0.5
                
                vote_margins.append(margin)
            
            # Sort margins and compare to target
            vote_margins.sort()
            target_margins_array = np.array(target_margins)
            
            # Calculate mean squared error between actual and target
            margins_error = np.mean((np.array(vote_margins) - target_margins_array) ** 2)
            score += weights['election_results'] * margins_error
        
        return score

    @staticmethod
    def _static_update_stats(district_stats, pixel_i, pixel_j, old_district, new_district, 
                            state_map, neighbor_map, height, width):
        """Static method to update district stats when a pixel changes district"""
        # Get the data for the pixel
        pixel_pop = state_map[pixel_i, pixel_j, 0]
        pixel_red = state_map[pixel_i, pixel_j, 1]
        pixel_blue = state_map[pixel_i, pixel_j, 2]
        
        # Update population and votes
        district_stats['population'][old_district] -= pixel_pop
        district_stats['population'][new_district] += pixel_pop
        
        district_stats['red_votes'][old_district] -= pixel_red
        district_stats['red_votes'][new_district] += pixel_red
        
        district_stats['blue_votes'][old_district] -= pixel_blue
        district_stats['blue_votes'][new_district] += pixel_blue
        
        # Update centers
        if district_stats['population'][old_district] > 0:
            old_total_pop = district_stats['population'][old_district] + pixel_pop
            old_center_x = district_stats['center_x'][old_district]
            old_center_y = district_stats['center_y'][old_district]
            
            district_stats['center_x'][old_district] = (old_center_x * old_total_pop - pixel_j * pixel_pop) / district_stats['population'][old_district]
            district_stats['center_y'][old_district] = (old_center_y * old_total_pop - pixel_i * pixel_pop) / district_stats['population'][old_district]
        
        # Add to new district center calculation
        new_total_pop = district_stats['population'][new_district]
        if new_total_pop > 0:
            old_new_pop = new_total_pop - pixel_pop
            if old_new_pop > 0:
                old_center_x = district_stats['center_x'][new_district]
                old_center_y = district_stats['center_y'][new_district]
                
                district_stats['center_x'][new_district] = (old_center_x * old_new_pop + pixel_j * pixel_pop) / new_total_pop
                district_stats['center_y'][new_district] = (old_center_y * old_new_pop + pixel_i * pixel_pop) / new_total_pop
            else:
                district_stats['center_x'][new_district] = pixel_j
                district_stats['center_y'][new_district] = pixel_i
        
        # Update area
        district_stats['area'][old_district] -= 1
        district_stats['area'][new_district] += 1
        
        # Simplified perimeter update for parallel processing
        # This is less accurate but faster for parallel runs
        neighbors = neighbor_map.get((pixel_i, pixel_j), [])
        old_perimeter_change = len(neighbors)
        new_perimeter_change = len(neighbors)
        
        district_stats['perimeter'][old_district] += old_perimeter_change
        district_stats['perimeter'][new_district] += new_perimeter_change
    
    def run_iteration(self):
        """Run a single iteration with improved acceptance criteria and aggressive population balancing"""
        # Get all boundary pixels
        boundary_pixels = self.get_boundary_pixels()
        
        if len(boundary_pixels) == 0:
            return False
        
        # Calculate mean population per district for targeting
        mean_population = np.mean(self.district_stats['population'])
        
        # Identify districts with population imbalance
        overpopulated_districts = np.where(self.district_stats['population'] > mean_population * 1.1)[0]
        underpopulated_districts = np.where(self.district_stats['population'] < mean_population * 0.9)[0]
        
        # Phase 1 logic: Population balancing 
        if self.phase == 1 and (len(overpopulated_districts) > 0 or len(underpopulated_districts) > 0):
            # In phase 1, direct population balancing by focusing on boundary pixels between
            # over and under populated districts
            targeted_boundary_pixels = []
            
            for pixel_i, pixel_j in boundary_pixels:
                old_district = self.district_map[pixel_i, pixel_j]
                
                # Check neighboring districts
                for ni, nj in self.neighbor_map.get((pixel_i, pixel_j), []):
                    new_district = self.district_map[ni, nj]
                    
                    # If this pixel is in an overpopulated district and neighbor is in underpopulated
                    if (old_district in overpopulated_districts and 
                        new_district in underpopulated_districts):
                        targeted_boundary_pixels.append((pixel_i, pixel_j, new_district))
                        break
            
            # If we found targeted pixels, use one of them
            if targeted_boundary_pixels:
                # Choose a random targeted pixel
                pixel_i, pixel_j, new_district = random.choice(targeted_boundary_pixels)
                old_district = self.district_map[pixel_i, pixel_j]
                
                # Check if this would break the district
                if not self.will_break_district(pixel_i, pixel_j, old_district):
                    # Make the change regardless of score in phase 1 (aggressive population balancing)
                    self.district_map[pixel_i, pixel_j] = new_district
                    self.update_district_stats(pixel_i, pixel_j, old_district, new_district)
                    return True
        
        # Standard approach for other phases or if targeted approach didn't work
        # Randomly select a boundary pixel
        idx = np.random.randint(0, len(boundary_pixels))
        pixel_i, pixel_j = boundary_pixels[idx]
        old_district = self.district_map[pixel_i, pixel_j]
        
        # Find a neighboring district
        neighboring_districts = set()
        for ni, nj in self.neighbor_map.get((pixel_i, pixel_j), []):
            neighboring_districts.add(self.district_map[ni, nj])
        
        neighboring_districts.discard(old_district)
        
        if not neighboring_districts:
            return False
        
        # In phase 1 and 2, prioritize moves that balance population
        if self.phase <= 1 and overpopulated_districts.size > 0 and underpopulated_districts.size > 0:
            preferred_districts = []
            
            # If we're in overpopulated district, prefer moving to underpopulated
            if old_district in overpopulated_districts:
                preferred_districts = [d for d in neighboring_districts if d in underpopulated_districts]
            
            # If we're in underpopulated, don't move to overpopulated
            elif old_district in underpopulated_districts:
                preferred_districts = [d for d in neighboring_districts if d not in overpopulated_districts]
            
            # Use preferred districts if available
            if preferred_districts:
                new_district = random.choice(preferred_districts)
            else:
                new_district = random.choice(list(neighboring_districts))
        else:
            new_district = random.choice(list(neighboring_districts))
        
        # Check if this would break the district
        if self.will_break_district(pixel_i, pixel_j, old_district):
            return False
        
        # Score the current map
        current_score = self.score_map()
        
        # Make the change and score again
        self.district_map[pixel_i, pixel_j] = new_district
        self.update_district_stats(pixel_i, pixel_j, old_district, new_district)
        new_score = self.score_map()
        
        # Decide whether to accept the change
        if new_score <= current_score:
            # Always accept improvements
            return True
        else:
            # For population balancing in phase 1, be more lenient
            if self.phase == 1 and old_district in overpopulated_districts and new_district in underpopulated_districts:
                # Accept with higher probability for population balancing
                accept_probability = 0.9
            else:
                # Score difference normalized by current score
                relative_diff = (new_score - current_score) / (current_score + 1e-10)
                accept_probability = np.exp(-relative_diff / max(self.temperature, 0.001))
            
            if random.random() < accept_probability:
                # Accept the change despite being worse
                return True
            else:
                # Revert the change
                self.district_map[pixel_i, pixel_j] = old_district
                self.update_district_stats(pixel_i, pixel_j, new_district, old_district)
                return False
    def update_phase(self, iteration, max_iterations):
        """Update the phase and adjust weights accordingly with improved phase transitions"""
        # Define phase transition points
        phase1_end = int(0.20 * max_iterations)  
        phase2_end = int(0.40 * max_iterations)  
        phase3_end = int(0.75 * max_iterations)  
        # Phase 4 - Final refinement (remaining 25%)
        
        # Determine current phase based on iteration
        if iteration == 0:
            # Phase 1: Focus almost exclusively on population equality
            self.phase = 1
            self.weights = {
                'population_equality': 3,  # Very high weight for population equality
                'compactness': 0,               # Low weight for compactness
                'center_distance': 1,           # Ignore center distance
                'election_results': 0           # Ignore election results initially
            }
            self.temperature = 0.7              # Start with max temperature to accept almost any change
            print(f"Iteration {iteration}: Phase 1 - Equalizing population")
            
        elif iteration == phase1_end:
            # Phase 2: Begin focusing on election results/vote distribution
            self.phase = 2
            self.weights = {
                'population_equality': 1,   # Still high but reduced
                'compactness': 2,               # Still low
                'center_distance': 2,           # Still ignored
                'election_results': 5        # Start optimizing for vote distribution
            }
            self.temperature = 0.6             # Still accepting many worse solutions
            print(f"Iteration {iteration}: Phase 2 - Optimizing vote distribution")
        
        elif iteration == phase2_end:
            # Phase 3: Focus on compactness while maintaining population equality and vote distribution
            self.phase = 3
            self.weights = {
                'population_equality': 1,    # Reduced but still important
                'compactness': 3,              # Increased focus on shape
                'center_distance': 3,          # Some focus on center distance
                'election_results': 2        # Decreased but still important
            }
            self.temperature = 0.3              # Less willing to accept worse solutions
            print(f"Iteration {iteration}: Phase 3 - Improving district compactness")
        
        elif iteration == phase3_end:
            # Phase 4: Final refinement
            self.phase = 4
            self.weights = {
                'population_equality': 1,    # Still important
                'compactness': 1,             # Very high focus on compactness
                'center_distance': 1,          # Higher focus on center distance
                'election_results': 2         # Still maintained but reduced
            }
            self.temperature = 0.1              # Only accept small increases in score
            print(f"Iteration {iteration}: Phase 4 - Final refinement")
    
    
    def _process_batch_multi(self, batch_size=100, pixels_per_move=20):
        """
        Process a batch of iterations with multiple pixels changed in each move
        
        Parameters:
        - batch_size: Number of moves to attempt
        - pixels_per_move: Number of pixels to change in each move
        
        Returns:
        - Number of accepted moves
        """
        accepted_count = 0
        
        # Get all boundary pixels - this is computationally expensive, so do it once per batch
        boundary_pixels = self.get_boundary_pixels()
        
        if len(boundary_pixels) == 0:
            return 0
        
        # Calculate mean population per district for targeting
        mean_population = np.mean(self.district_stats['population'])
        
        # Identify districts with population imbalance
        overpopulated_districts = np.where(self.district_stats['population'] > mean_population * 1.1)[0]
        underpopulated_districts = np.where(self.district_stats['population'] < mean_population * 0.9)[0]
        
        for _ in range(batch_size):
            # Phase 1: Population balancing - focus on problematic districts
            if self.phase == 1 and (len(overpopulated_districts) > 0 and len(underpopulated_districts) > 0):
                # In phase 1, direct population balancing by focusing on boundary pixels between
                # over and under populated districts
                targeted_boundary_pixels = []
                
                for pixel_i, pixel_j in boundary_pixels:
                    old_district = self.district_map[pixel_i, pixel_j]
                    
                    # Only consider pixels in overpopulated districts
                    if old_district not in overpopulated_districts:
                        continue
                        
                    # Check neighboring districts
                    for ni, nj in self.neighbor_map.get((pixel_i, pixel_j), []):
                        new_district = self.district_map[ni, nj]
                        
                        # If neighbor is in underpopulated district
                        if new_district in underpopulated_districts:
                            # Check if moving would break the district
                            if not self.will_break_district(pixel_i, pixel_j, old_district):
                                targeted_boundary_pixels.append((pixel_i, pixel_j, old_district, new_district))
                
                # If we found targeted pixels, use them
                if len(targeted_boundary_pixels) > 0:
                    # Pick up to pixels_per_move pixels, or as many as we found
                    num_to_change = min(pixels_per_move, len(targeted_boundary_pixels))
                    selected_moves = random.sample(targeted_boundary_pixels, num_to_change)
                    
                    # Make all changes at once
                    for pixel_i, pixel_j, old_district, new_district in selected_moves:
                        self.district_map[pixel_i, pixel_j] = new_district
                        self.update_district_stats(pixel_i, pixel_j, old_district, new_district)
                    
                    accepted_count += 1
                    continue
            
            # For other phases or if targeted approach didn't work
            # Try random boundary pixels with standard evaluation
            
            # First check if we have enough boundary pixels
            if len(boundary_pixels) < pixels_per_move:
                # Not enough pixels, use what we have
                num_to_try = len(boundary_pixels)
            else:
                num_to_try = pixels_per_move
            
            # Choose random boundary pixels
            random_indices = np.random.choice(len(boundary_pixels), num_to_try, replace=False)
            
            # Prepare the moves
            proposed_moves = []
            
            for idx in random_indices:
                pixel_i, pixel_j = boundary_pixels[idx]
                old_district = self.district_map[pixel_i, pixel_j]
                
                # Find a neighboring district
                neighboring_districts = set()
                for ni, nj in self.neighbor_map.get((pixel_i, pixel_j), []):
                    neighboring_districts.add(self.district_map[ni, nj])
                
                neighboring_districts.discard(old_district)
                
                if not neighboring_districts:
                    continue
                
                # In phase 1 and 2, prioritize moves that balance population
                if self.phase <= 2 and overpopulated_districts.size > 0 and underpopulated_districts.size > 0:
                    preferred_districts = []
                    
                    # If we're in overpopulated district, prefer moving to underpopulated
                    if old_district in overpopulated_districts:
                        preferred_districts = [d for d in neighboring_districts if d in underpopulated_districts]
                    
                    # If we're in underpopulated, don't move to overpopulated
                    elif old_district in underpopulated_districts:
                        preferred_districts = [d for d in neighboring_districts if d not in overpopulated_districts]
                    
                    # Use preferred districts if available
                    if preferred_districts:
                        new_district = random.choice(preferred_districts)
                    else:
                        new_district = random.choice(list(neighboring_districts))
                else:
                    new_district = random.choice(list(neighboring_districts))
                
                # Check if this would break the district
                if not self.will_break_district(pixel_i, pixel_j, old_district):
                    proposed_moves.append((pixel_i, pixel_j, old_district, new_district))
            
            # If no valid moves, continue to next iteration
            if not proposed_moves:
                continue
            
            # Score the current map
            current_score = self.score_map()
            
            # Make all the changes at once
            for pixel_i, pixel_j, old_district, new_district in proposed_moves:
                self.district_map[pixel_i, pixel_j] = new_district
                self.update_district_stats(pixel_i, pixel_j, old_district, new_district)
            
            # Score the new map
            new_score = self.score_map()
            
            # Decide whether to accept all changes
            if new_score <= current_score:
                # Accept all changes (they improved the score)
                accepted_count += 1
            else:
                # For population balancing in phase 1, be more lenient
                if self.phase == 1 and any(old_district in overpopulated_districts and new_district in underpopulated_districts 
                                        for _, _, old_district, new_district in proposed_moves):
                    # Accept with higher probability for population balancing
                    accept_probability = 0.9
                else:
                    # Use relative score difference
                    relative_diff = (new_score - current_score) / (current_score + 1e-10)
                    accept_probability = np.exp(-relative_diff / max(self.temperature, 0.001))
                
                if random.random() < accept_probability:
                    # Accept all changes despite being worse
                    accepted_count += 1
                else:
                    # Revert all changes
                    for pixel_i, pixel_j, old_district, new_district in proposed_moves:
                        self.district_map[pixel_i, pixel_j] = old_district
                        self.update_district_stats(pixel_i, pixel_j, new_district, old_district)
        
        return accepted_count

    def run_simulation(self, num_iterations=100000, batch_size=1000, use_parallel=True, pixels_per_move=20):
        """
        Run the simulation with multiple pixels changed per move
        
        Parameters:
        - num_iterations: Total number of iterations (moves) to run
        - batch_size: Number of moves per batch for parallel processing
        - use_parallel: Whether to use parallel processing
        - pixels_per_move: Number of pixels to change in each move
        """
        # Initialize the phase for iteration 0
        self.update_phase(0, num_iterations)
        
        if use_parallel and self.num_cpus > 1:
            print(f"Running simulation using {self.num_cpus} CPU cores in parallel with {pixels_per_move} pixels per move")
            iterations_completed = 0
            progress_bar = tqdm(total=num_iterations)
            
            while iterations_completed < num_iterations:
                # Determine current batch size
                current_batch_size = min(batch_size, num_iterations - iterations_completed)
                
                # Run batch with multiple pixels per move for early phases
                if self.phase <= 2:
                    accepted = self._process_batch_multi(batch_size=current_batch_size, pixels_per_move=pixels_per_move)
                else:
                    # For later phases focusing on compactness, use parallel processing with single pixel moves
                    accepted = self.run_batch_parallel(batch_size=current_batch_size)
                
                # Update iteration count
                iterations_completed += current_batch_size
                
                # Update phase
                self.update_phase(iterations_completed, num_iterations)
                
                # Update progress bar
                progress_bar.update(current_batch_size)
                
                # Report progress more frequently
                if iterations_completed % 1000 == 0:
                    # Calculate statistics
                    self.calculate_all_district_stats()
                    
                    # Get population information
                    pop_mean = np.mean(self.district_stats['population'])
                    pop_max = np.max(self.district_stats['population'])
                    pop_min = np.min(self.district_stats['population'])
                    pop_imbalance = (pop_max - pop_min) / pop_mean
                    
                    # Get district counts
                    red_districts = sum(1 for d in range(self.num_districts) 
                                if self.district_stats['red_votes'][d] > 
                                    self.district_stats['blue_votes'][d])
                    blue_districts = self.num_districts - red_districts
                    
                    # Output status
                    current_score = self.score_map()
                    print(f"\nIteration {iterations_completed}/{num_iterations}, Score: {current_score:.2f}")
                    print(f"Population imbalance: {pop_imbalance:.2%}, Min: {pop_min:.0f}, Max: {pop_max:.0f}")
                    print(f"Districts: {red_districts} Red, {blue_districts} Blue")
                    print(f"Temperature: {self.temperature:.4f}, Phase: {self.phase}")
            
            progress_bar.close()
            
            # Clean up
            if self.pool:
                self.pool.close()
                self.pool.join()
                self.pool = None
        else:
            # Implement single-threaded version with multi-pixel moves
            print(f"Running simulation in single-threaded mode with {pixels_per_move} pixels per move")
            progress_bar = tqdm(total=num_iterations)
            
            for i in range(0, num_iterations, pixels_per_move):
                # For early phases focusing on population balancing, use multi-pixel moves
                if self.phase <= 2:
                    self._process_batch_multi(batch_size=1, pixels_per_move=pixels_per_move)
                else:
                    # For later phases, use single pixel moves
                    for _ in range(pixels_per_move):
                        self.run_iteration()
                
                # Update iteration count
                current_iteration = min(i + pixels_per_move, num_iterations)
                
                # Update phase
                self.update_phase(current_iteration, num_iterations)
                
                # Update progress every 1000 iterations
                if current_iteration % 1000 < pixels_per_move:
                    progress_bar.update(min(1000, current_iteration - progress_bar.n))
                    
                    # Report detailed status every 5000 iterations
                    if current_iteration % 5000 < pixels_per_move or current_iteration == num_iterations:
                        # Calculate statistics
                        self.calculate_all_district_stats()
                        
                        # Get population information
                        pop_mean = np.mean(self.district_stats['population'])
                        pop_max = np.max(self.district_stats['population'])
                        pop_min = np.min(self.district_stats['population'])
                        pop_imbalance = (pop_max - pop_min) / pop_mean
                        
                        # Get district counts
                        red_districts = sum(1 for d in range(self.num_districts) 
                                    if self.district_stats['red_votes'][d] > 
                                        self.district_stats['blue_votes'][d])
                        blue_districts = self.num_districts - red_districts
                        
                        # Output status
                        current_score = self.score_map()
                        print(f"\nIteration {current_iteration}/{num_iterations}, Score: {current_score:.2f}")
                        print(f"Population imbalance: {pop_imbalance:.2%}, Min: {pop_min:.0f}, Max: {pop_max:.0f}")
                        print(f"Districts: {red_districts} Red, {blue_districts} Blue")
                        print(f"Temperature: {self.temperature:.4f}, Phase: {self.phase}")
            
            progress_bar.close()
        
        # Calculate final statistics
        self.calculate_all_district_stats()
        print("Simulation complete!")
    
    def plot_districts(self, ax=None, show_stats=True):
        """Plot the current district map with a distinct background color"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a masked array where invalid areas are masked
        masked_district_map = np.ma.masked_array(
            self.district_map, 
            mask=~self.valid_mask
        )
        
        # Create a colormap for the districts with a distinct background
        cmap = plt.cm.get_cmap('tab20', self.num_districts)
        
        # Set a distinct background color (e.g., light gray)
        cmap.set_bad('white')
        
        # Plot the district map
        ax.imshow(masked_district_map, cmap=cmap)
        
        if show_stats:
            # Calculate election results
            district_winners = []
            vote_margins = []
            
            for district_id in range(self.num_districts):
                red = self.district_stats['red_votes'][district_id]
                blue = self.district_stats['blue_votes'][district_id]
                
                if red > blue:
                    district_winners.append('Red')
                else:
                    district_winners.append('Blue')
                
                total = red + blue
                if total > 0:
                    margin = red / total
                else:
                    margin = 0.5
                
                vote_margins.append(margin)
            
            red_seats = district_winners.count('Red')
            blue_seats = district_winners.count('Blue')
            
            total_red = np.sum(self.state_map[:,:,1])
            total_blue = np.sum(self.state_map[:,:,2])
            red_vote_pct = total_red / (total_red + total_blue) * 100
            
            # Show the results on the plot
            ax.set_title(f'Districts: {red_seats} Red, {blue_seats} Blue (Popular vote: {red_vote_pct:.1f}% Red)')
        
        ax.axis('off')
        return ax
    
    def plot_election_results(self, ax=None):
        """Plot the election results by district"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate vote margins
        vote_margins = []
        
        for district_id in range(self.num_districts):
            red = self.district_stats['red_votes'][district_id]
            blue = self.district_stats['blue_votes'][district_id]
            total = red + blue
            
            if total > 0:
                margin = red / total
            else:
                margin = 0.5
            
            vote_margins.append(margin)
        
        # Sort margins
        vote_margins.sort()
        
        # Plot actual margins
        district_indices = np.arange(1, self.num_districts + 1)
        ax.scatter(district_indices, vote_margins, color='red', label='Actual')
        
        # If we have target margins, plot those too
        if self.target_vote_margins is not None:
            ax.plot(district_indices, self.target_vote_margins, 'b--', label='Target')
        
        # Add a line at 50%
        ax.axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('District (Ranked by Republican vote share)')
        ax.set_ylabel('Republican vote share')
        ax.set_ylim(0, 1)
        ax.set_xlim(0.5, self.num_districts + 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def plot_metrics(self):
        """Plot various metrics for the current map"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot population by district
        ax = axes[0, 0]
        ax.bar(range(1, self.num_districts + 1), self.district_stats['population'])
        ax.set_xlabel('District')
        ax.set_ylabel('Population')
        ax.set_title('Population by District')
        
        # Plot perimeter to area ratios
        ax = axes[0, 1]
        perimeter_to_area = self.district_stats['perimeter'] / np.sqrt(self.district_stats['area'])
        ax.bar(range(1, self.num_districts + 1), perimeter_to_area)
        ax.set_xlabel('District')
        ax.set_ylabel('Perimeter / √Area')
        ax.set_title('Compactness by District')
        
        # Plot district map
        ax = axes[1, 0]
        self.plot_districts(ax)
        
        # Plot election results
        ax = axes[1, 1]
        self.plot_election_results(ax)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_district_map(self, output_file="district_map.npy"):
        """Save the district map to a file"""
        np.save(output_file, self.district_map)
        print(f"District map saved to {output_file}")
    
    def export_district_stats(self, output_file="district_stats.csv"):
        """Export district statistics to CSV file"""
        stats_df = pd.DataFrame({
            'district_id': range(self.num_districts),
            'population': self.district_stats['population'],
            'red_votes': self.district_stats['red_votes'],
            'blue_votes': self.district_stats['blue_votes'],
            'area': self.district_stats['area'],
            'perimeter': self.district_stats['perimeter'],
            'center_x': self.district_stats['center_x'],
            'center_y': self.district_stats['center_y']
        })
        
        # Calculate additional metrics
        stats_df['compactness'] = stats_df['perimeter'] / np.sqrt(stats_df['area'])
        stats_df['vote_margin'] = stats_df['red_votes'] / (stats_df['red_votes'] + stats_df['blue_votes'])
        stats_df['winner'] = stats_df['vote_margin'].apply(lambda x: 'Republican' if x > 0.5 else 'Democratic')
        
        # Save to CSV
        stats_df.to_csv(output_file, index=False)
        print(f"District statistics saved to {output_file}")
        