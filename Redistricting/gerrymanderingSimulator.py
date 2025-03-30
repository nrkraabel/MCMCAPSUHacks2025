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
import cupy as cp
import dask_cuda
import dask.array as da
from dask.distributed import Client, wait, get_worker
import time

class GerrymanderSimulator:
    def __init__(self, state_map, num_districts=13, use_gpu=True, num_gpus=None):
        """
        Initialize the simulator with a state map and number of districts.
        
        Parameters:
        - state_map: A numpy array with shape (height, width, 3) where each pixel has
                    [population, red_votes, blue_votes]
        - num_districts: Number of districts to create
        - use_gpu: Whether to use GPU acceleration
        - num_gpus: Number of GPUs to use (None = use all available)
        """
        self.state_map = state_map
        self.height, self.width, _ = state_map.shape
        self.num_districts = num_districts
        
        # Determine GPU availability and count
        self.use_gpu = use_gpu and cuda.is_available()
        if self.use_gpu:
            # Get number of available GPUs
            self.num_gpus = num_gpus if num_gpus is not None else cp.cuda.runtime.getDeviceCount()
            print(f"Found {self.num_gpus} CUDA-capable GPUs")
            
            if self.num_gpus > 1:
                # Initialize Dask CUDA cluster for multi-GPU processing
                try:
                    self.cluster = dask_cuda.LocalCUDACluster(n_workers=self.num_gpus, threads_per_worker=1)
                    self.client = Client(self.cluster)
                    print(f"Initialized Dask CUDA cluster with {self.num_gpus} workers")
                    self.using_dask = True
                except Exception as e:
                    print(f"Failed to initialize Dask CUDA cluster: {e}")
                    print("Falling back to single GPU mode")
                    self.num_gpus = 1
                    self.using_dask = False
            else:
                self.using_dask = False
                
            # Create GPU device contexts
            self.devices = [cp.cuda.Device(i) for i in range(self.num_gpus)]
            
            # Partition the map for multi-GPU processing
            self.partition_boundaries = self._calculate_partition_boundaries()
            
            # Initialize GPU arrays
            if self.using_dask:
                # Create Dask arrays distributed across GPUs
                state_map_shape = (self.height, self.width, 3)
                self.state_map_gpu = da.from_array(state_map, chunks=(self.height // self.num_gpus, self.width, 3))
                self.district_map = da.zeros((self.height, self.width), dtype=np.int32, 
                                           chunks=(self.height // self.num_gpus, self.width))
                # Valid mask as dask array
                self.valid_mask = self.state_map_gpu[:,:,0] > 0
            else:
                # Single GPU mode
                self.state_map_gpu = cp.asarray(state_map)
                self.district_map = cp.zeros((self.height, self.width), dtype=cp.int32)
                self.valid_mask = (self.state_map_gpu[:,:,0] > 0)
        else:
            print("Using CPU processing")
            self.num_gpus = 0
            self.district_map = np.zeros((self.height, self.width), dtype=np.int32)
            self.valid_mask = (state_map[:,:,0] > 0)
            
        # CPU resources
        self.num_cpus = max(1, os.cpu_count() - 2)  # Reserve just 2 cores for system
        print(f"Using {self.num_cpus} CPU cores for parallelization")
        
        # Initialize the district map randomly using Voronoi tessellation
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
        
        # If GPU is available, create GPU versions of these arrays
        if self.use_gpu:
            self.district_stats_gpu = {
                'population': cp.zeros(num_districts),
                'red_votes': cp.zeros(num_districts),
                'blue_votes': cp.zeros(num_districts),
                'center_x': cp.zeros(num_districts),
                'center_y': cp.zeros(num_districts),
                'perimeter': cp.zeros(num_districts),
                'area': cp.zeros(num_districts)
            }
            
            # For multi-GPU, we need per-device stats that will be merged
            if self.num_gpus > 1:
                self.device_stats = [{
                    'population': cp.zeros(num_districts),
                    'red_votes': cp.zeros(num_districts),
                    'blue_votes': cp.zeros(num_districts),
                    'center_x': cp.zeros(num_districts),
                    'center_y': cp.zeros(num_districts),
                    'perimeter': cp.zeros(num_districts),
                    'area': cp.zeros(num_districts)
                } for _ in range(self.num_gpus)]
        
        # Calculate initial stats
        self.calculate_all_district_stats()
        
        # Parameters for the algorithm
        self.temperature = 1
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
        
        # Create a pool of workers for parallelization when CPU is needed
        self.pool = None  # Will initialize when needed
    
    def _calculate_partition_boundaries(self):
        """Calculate boundaries for partitioning the state map across multiple GPUs"""
        if self.num_gpus <= 1:
            return [(0, self.height)]
            
        # For simplicity, partition by rows (can be optimized further)
        rows_per_gpu = self.height // self.num_gpus
        boundaries = []
        
        for i in range(self.num_gpus):
            start_row = i * rows_per_gpu
            end_row = start_row + rows_per_gpu if i < self.num_gpus - 1 else self.height
            boundaries.append((start_row, end_row))
            
        return boundaries
    
    def _precompute_neighbor_map(self):
        """Precompute the neighbor map for faster lookup"""
        self.neighbor_map = {}
        
        # Decide whether to use GPU or CPU operations
        if self.use_gpu:
            if self.using_dask:
                # For Dask distributed computing, we need to compute the valid mask
                valid_mask = self.valid_mask.compute()
            else:
                # Get valid mask as numpy array for iteration
                valid_mask = cp.asnumpy(self.valid_mask)
        else:
            valid_mask = self.valid_mask
        
        # For each valid cell, store its neighbors
        for i in range(self.height):
            for j in range(self.width):
                if valid_mask[i, j]:
                    self.neighbor_map[(i, j)] = []
                    for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                        if 0 <= ni < self.height and 0 <= nj < self.width and valid_mask[ni, nj]:
                            self.neighbor_map[(i, j)].append((ni, nj))
    
    def initialize_districts(self):
        """Initialize district map using Voronoi tessellation"""
        # Generate random seed points
        if self.use_gpu:
            if self.using_dask:
                valid_mask_np = self.valid_mask.compute()
                valid_indices = np.argwhere(valid_mask_np)
            else:
                valid_indices = cp.argwhere(self.valid_mask).get()  # Get from GPU to CPU for Voronoi
        else:
            valid_indices = np.argwhere(self.valid_mask)
        
        seed_indices = valid_indices[np.random.choice(len(valid_indices), self.num_districts, replace=False)]
        
        # Create a Voronoi diagram
        vor = Voronoi(seed_indices)
        
        # Multi-GPU optimized assignment
        if self.use_gpu and self.num_gpus > 1:
            # For multi-GPU case, we'll assign each partition to a specific GPU
            futures = []
            
            for gpu_id, (start_row, end_row) in enumerate(self.partition_boundaries):
                # Submit a task to each GPU
                futures.append(self.client.submit(
                    self._initialize_district_partition,
                    gpu_id, start_row, end_row, self.width, self.valid_mask, seed_indices,
                    resources={'GPU': 1}  # Ensure task is assigned to a GPU worker
                ))
            
            # Wait for all partitions to complete
            results = self.client.gather(futures)
            
            # Combine results from all GPUs
            if self.using_dask:
                # For Dask, we need to update our dask array
                for gpu_id, partition_map in enumerate(results):
                    start_row, end_row = self.partition_boundaries[gpu_id]
                    partition_dask = da.from_array(partition_map, 
                                                chunks=(end_row-start_row, self.width))
                    self.district_map[start_row:end_row] = partition_dask
            else:
                # For single GPU, just use CuPy arrays
                self.district_map = cp.asarray(np.vstack([r for r in results]))
        
        elif self.use_gpu:
            # Single GPU case
            # Define CUDA kernel for pixel assignment
            @cuda.jit
            def assign_pixels_kernel(district_map, valid_mask, seed_indices, height, width):
                # Get thread position
                i, j = cuda.grid(2)
                
                # Check if in bounds and valid
                if i < height and j < width and valid_mask[i, j]:
                    # Find the closest seed point
                    min_dist = 1e10  # A large value
                    closest_idx = 0
                    
                    for idx in range(len(seed_indices)):
                        seed_i, seed_j = seed_indices[idx]
                        dist = (i - seed_i)**2 + (j - seed_j)**2
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = idx
                    
                    district_map[i, j] = closest_idx
            
            # Convert seed indices to device array
            d_seed_indices = cuda.to_device(seed_indices)
            
            # Set up grid and block dimensions
            threads_per_block = (16, 16)
            blocks_per_grid_x = (self.height + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (self.width + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            # Get numpy arrays from cupy for CUDA
            d_district_map = cuda.to_device(cp.zeros((self.height, self.width), dtype=cp.int32).get())
            d_valid_mask = cuda.to_device(cp.asnumpy(self.valid_mask))
            
            # Launch kernel
            assign_pixels_kernel[blocks_per_grid, threads_per_block](
                d_district_map, d_valid_mask, d_seed_indices, self.height, self.width)
            
            # Copy result back
            self.district_map = cp.asarray(d_district_map.copy_to_host())
        else:
            # Use the existing CPU implementation with Numba
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
    
    @staticmethod
    def _initialize_district_partition(gpu_id, start_row, end_row, width, valid_mask, seed_indices):
        """Initialize a partition of the district map on a specific GPU"""
        # Set the GPU device
        with cp.cuda.Device(gpu_id):
            # Extract the partition of valid_mask for this GPU
            if isinstance(valid_mask, da.Array):
                # For Dask arrays, compute the slice
                partition_mask = valid_mask[start_row:end_row].compute()
            else:
                # For NumPy arrays
                partition_mask = valid_mask[start_row:end_row]
                
            # Create a new array for this partition
            partition_height = end_row - start_row
            partition_map = cp.zeros((partition_height, width), dtype=cp.int32)
            
            # Define CUDA kernel for assigning districts
            @cuda.jit
            def assign_pixels_kernel(district_map, valid_mask, seed_indices, height, width, start_row):
                # Get thread position within the partition
                local_i, j = cuda.grid(2)
                
                # Convert to global coordinates
                i = local_i + start_row
                
                # Check if in bounds and valid
                if local_i < height and j < width and valid_mask[local_i, j]:
                    # Find the closest seed point
                    min_dist = 1e10  # A large value
                    closest_idx = 0
                    
                    for idx in range(len(seed_indices)):
                        seed_i, seed_j = seed_indices[idx]
                        dist = (i - seed_i)**2 + (j - seed_j)**2
                        if dist < min_dist:
                            min_dist = dist
                            closest_idx = idx
                    
                    district_map[local_i, j] = closest_idx
            
            # Convert to CUDA
            d_seed_indices = cuda.to_device(seed_indices)
            d_district_map = cuda.to_device(cp.asnumpy(partition_map))
            d_valid_mask = cuda.to_device(cp.asnumpy(partition_mask))
            
            # Set up grid and block dimensions
            threads_per_block = (16, 16)
            blocks_per_grid_x = (partition_height + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (width + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            # Launch kernel
            assign_pixels_kernel[blocks_per_grid, threads_per_block](
                d_district_map, d_valid_mask, d_seed_indices, partition_height, width, start_row)
            
            # Get results back
            partition_result = d_district_map.copy_to_host()
            
            # Return the partition
            return partition_result
    
    def _fill_holes(self):
        """
        Fill in small holes in the district map.
        A hole is defined as a zero-population area completely surrounded by districts.
        """
        # If using GPU, bring data to CPU for hole filling operations
        if self.use_gpu:
            if self.using_dask:
                district_map_np = self.district_map.compute()
                valid_mask_np = self.valid_mask.compute()
                state_map_np = self.state_map_gpu.compute()
            else:
                district_map_np = cp.asnumpy(self.district_map)
                valid_mask_np = cp.asnumpy(self.valid_mask)
                state_map_np = cp.asnumpy(self.state_map_gpu)
        else:
            district_map_np = self.district_map
            valid_mask_np = self.valid_mask
            state_map_np = self.state_map
            
        # Create a mask of areas that have been assigned to districts
        district_assigned = (district_map_np >= 0) & valid_mask_np
        
        # Identify potential holes (zero-population areas)
        potential_holes = (state_map_np[:,:,0] == 0) & ~district_assigned
        
        # If there are no potential holes, return early
        if not np.any(potential_holes):
            if self.use_gpu:
                # Update GPU arrays if needed
                if self.using_dask:
                    self.district_map = da.from_array(district_map_np, 
                                              chunks=(self.height // self.num_gpus, self.width))
                    self.valid_mask = da.from_array(valid_mask_np,
                                            chunks=(self.height // self.num_gpus, self.width))
                else:
                    self.district_map = cp.asarray(district_map_np)
                    self.valid_mask = cp.asarray(valid_mask_np)
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
            surrounding_districts = district_map_np[neighbors_mask]
            valid_neighbors = np.sum(valid_mask_np[neighbors_mask])
            
            # If most of the surrounding pixels are assigned to districts, this is a hole
            if valid_neighbors > 0 and valid_neighbors / np.sum(neighbors_mask) > 0.5:
                print(f"Filling hole with size {hole_size}")
                
                # Get list of surrounding districts
                neighbor_districts = []
                for i, j in np.argwhere(hole_mask):
                    for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                        if (0 <= ni < self.height and 0 <= nj < self.width and 
                            valid_mask_np[ni, nj]):
                            neighbor_districts.append(district_map_np[ni, nj])
                
                # Find most common neighboring district
                if neighbor_districts:
                    from collections import Counter
                    most_common_district = Counter(neighbor_districts).most_common(1)[0][0]
                    
                    # Fill the hole with this district
                    district_map_np[hole_mask] = most_common_district
                    valid_mask_np[hole_mask] = True
        
        # Update the instance variables with processed data
        if self.use_gpu:
            if self.using_dask:
                self.district_map = da.from_array(district_map_np, 
                                               chunks=(self.height // self.num_gpus, self.width))
                self.valid_mask = da.from_array(valid_mask_np,
                                             chunks=(self.height // self.num_gpus, self.width))
            else:
                self.district_map = cp.asarray(district_map_np)
                self.valid_mask = cp.asarray(valid_mask_np)
        else:
            self.district_map = district_map_np
            self.valid_mask = valid_mask_np
    
    def calculate_all_district_stats(self):
        """Calculate all statistics for all districts with multi-GPU support"""
        # Reset stats
        for key in self.district_stats:
            self.district_stats[key] = np.zeros(self.num_districts)
            
        if self.use_gpu:
            for key in self.district_stats_gpu:
                self.district_stats_gpu[key] = cp.zeros(self.num_districts)
            
            if self.num_gpus > 1 and self.using_dask:
                # Multi-GPU approach using Dask
                futures = []
                
                # Reset per-device stats
                for d in range(self.num_gpus):
                    for key in self.device_stats[d]:
                        with cp.cuda.Device(d):
                            self.device_stats[d][key] = cp.zeros(self.num_districts)
                
                # Submit calculation tasks to each GPU for its partition
                for gpu_id, (start_row, end_row) in enumerate(self.partition_boundaries):
                    futures.append(self.client.submit(
                        self._calculate_partition_stats,
                        gpu_id, start_row, end_row, self.num_districts,
                        resources={'GPU': 1}
                    ))
                
                # Gather results
                device_stats_results = self.client.gather(futures)
                
                # Combine stats from all devices
                for result in device_stats_results:
                    for key in self.district_stats_gpu:
                        self.district_stats_gpu[key] += result[key]
                
                # Copy results to CPU
                for key in self.district_stats:
                    self.district_stats[key] = cp.asnumpy(self.district_stats_gpu[key])
                
            elif self.num_gpus > 1:
                # Multi-GPU without Dask (manual management)
                # Reset per-device stats
                for d in range(self.num_gpus):
                    for key in self.device_stats[d]:
                        with cp.cuda.Device(d):
                            self.device_stats[d][key] = cp.zeros(self.num_districts)
                
                # Process each partition on its assigned GPU
                for gpu_id, (start_row, end_row) in enumerate(self.partition_boundaries):
                    with cp.cuda.Device(gpu_id):
                        # Extract the partition for this GPU
                        partition_map = self.district_map[start_row:end_row]
                        partition_state = self.state_map_gpu[start_row:end_row]
                        
                        # Calculate stats for this partition
                        district_ids = cp.unique(partition_map)
                        
                        for district_id in district_ids.get():
                            mask = (partition_map == district_id)
                            
                            # Count population and votes using vectorized operations
                            self.device_stats[gpu_id]['population'][district_id] += cp.sum(partition_state[:,:,0] * mask)
                            self.device_stats[gpu_id]['red_votes'][district_id] += cp.sum(partition_state[:,:,1] * mask)
                            self.device_stats[gpu_id]['blue_votes'][district_id] += cp.sum(partition_state[:,:,2] * mask)
                            
                            # Center calculations will be adjusted later
                            pop_indices = cp.argwhere(mask & (partition_state[:,:,0] > 0))
                            if len(pop_indices) > 0:
                                pop_weights = cp.array([partition_state[i, j, 0] for i, j in pop_indices])
                                # Adjust i coordinate by start_row for global position
                                adjusted_indices = pop_indices.copy()
                                adjusted_indices[:, 0] += start_row
                                
                                # Accumulate weighted coordinates
                                self.device_stats[gpu_id]['center_y'][district_id] += cp.sum(adjusted_indices[:, 0] * pop_weights)
                                self.device_stats[gpu_id]['center_x'][district_id] += cp.sum(adjusted_indices[:, 1] * pop_weights)
                            
                            # Area is the number of pixels in this district in this partition
                            self.device_stats[gpu_id]['area'][district_id] = cp.sum(mask)
                            
                            # Calculate perimeter (simplified for now)
                            self.device_stats[gpu_id]['perimeter'][district_id] = self._calculate_perimeter_gpu(district_id, gpu_id, start_row, end_row)
                
                # Combine stats from all devices
                for key in self.district_stats_gpu:
                    self.district_stats_gpu[key] = cp.zeros(self.num_districts)
                    for d in range(self.num_gpus):
                        with cp.cuda.Device(0):  # Use device 0 for final combination
                            self.district_stats_gpu[key] += cp.asarray(self.device_stats[d][key].get())
                
                # Adjust center calculations
                for district_id in range(self.num_districts):
                    if self.district_stats_gpu['population'][district_id] > 0:
                        self.district_stats_gpu['center_y'][district_id] /= self.district_stats_gpu['population'][district_id]
                        self.district_stats_gpu['center_x'][district_id] /= self.district_stats_gpu['population'][district_id]
                
                # Copy to CPU
                for key in self.district_stats:
                    self.district_stats[key] = cp.asnumpy(self.district_stats_gpu[key])
                
            else:
                # Single GPU implementation
                # Use GPU operations for faster calculation
                if self.using_dask:
                    # For Dask arrays, compute to get concrete arrays
                    district_map_cp = cp.asarray(self.district_map.compute())
                    valid_mask_cp = cp.asarray(self.valid_mask.compute())
                    state_map_cp = cp.asarray(self.state_map_gpu.compute())
                else:
                    district_map_cp = self.district_map
                    valid_mask_cp = self.valid_mask
                    state_map_cp = self.state_map_gpu
                
                district_ids = cp.unique(district_map_cp[valid_mask_cp])
                
                for district_id in district_ids.get():  # Iterate over NumPy array
                    mask = (district_map_cp == district_id)
                    
                    # Count population and votes using vectorized operations
                    self.district_stats_gpu['population'][district_id] = cp.sum(state_map_cp[:,:,0] * mask)
                    self.district_stats_gpu['red_votes'][district_id] = cp.sum(state_map_cp[:,:,1] * mask)
                    self.district_stats_gpu['blue_votes'][district_id] = cp.sum(state_map_cp[:,:,2] * mask)
                    
                    # Calculate center of population
                    if self.district_stats_gpu['population'][district_id] > 0:
                        # Get indices where the mask is True
                        pop_indices = cp.argwhere(mask & (state_map_cp[:,:,0] > 0))
                        if len(pop_indices) > 0:
                            pop_weights = cp.array([state_map_cp[i, j, 0] for i, j in pop_indices])
                            
                            self.district_stats_gpu['center_y'][district_id] = cp.average(pop_indices[:, 0], weights=pop_weights)
                            self.district_stats_gpu['center_x'][district_id] = cp.average(pop_indices[:, 1], weights=pop_weights)
                    
                    # For perimeter and area, bring data to CPU for calculation
                    district_pixels = cp.argwhere(mask).get()
                    self.district_stats_gpu['area'][district_id] = len(district_pixels)
                    
                    # Calculate perimeter efficiently using CUDA kernel
                    perimeter = self._calculate_perimeter_gpu(district_id)
                    self.district_stats_gpu['perimeter'][district_id] = perimeter
                
                # Copy data from GPU to CPU
                for key in self.district_stats:
                    self.district_stats[key] = cp.asnumpy(self.district_stats_gpu[key])
        else:
            # Use the original CPU implementation
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

    def _will_break_district_partition(self, partition_map, local_i, j, district_id):
        """Check if removing a pixel would break district connectivity in this partition"""
        # Convert to CPU for numba
        partition_map_np = cp.asnumpy(partition_map)
        
        @jit(nopython=True)
        def check_connectivity(map_data, i, j, district, height, width):
            # Get neighbors of the same district
            neighbors = []
            for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                if 0 <= ni < height and 0 <= nj < width and map_data[ni, nj] == district:
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
                        if 0 <= ni < height and 0 <= nj < width and not visited[ni, nj] and map_data[ni, nj] == district:
                            if ni == i and nj == j:
                                continue  # Skip the pixel we're removing
                            visited[ni, nj] = True
                            queue.append((ni, nj))
                
                # Check if all neighbors were reached
                for neighbor in neighbors[1:]:
                    if not visited[neighbor[0], neighbor[1]]:
                        return True  # District would be broken
            
            return False
        
        # Use Numba function
        height, width = partition_map_np.shape
        return check_connectivity(partition_map_np, local_i, j, district_id, height, width)
    
    def _update_stats_partition(self, partition_stats, partition_state, local_i, j, old_district, new_district, start_row):
        """Update district stats when a pixel changes district in a partition"""
        # Global i coordinate
        i = local_i + start_row
        
        # Get the data for the pixel
        pixel_pop = float(partition_state[local_i, j, 0])
        pixel_red = float(partition_state[local_i, j, 1])
        pixel_blue = float(partition_state[local_i, j, 2])
        
        # Update population and votes
        partition_stats['population'][old_district] -= pixel_pop
        partition_stats['population'][new_district] += pixel_pop
        
        partition_stats['red_votes'][old_district] -= pixel_red
        partition_stats['red_votes'][new_district] += pixel_red
        
        partition_stats['blue_votes'][old_district] -= pixel_blue
        partition_stats['blue_votes'][new_district] += pixel_blue
        
        # Update centers using the formula for adding/removing from weighted average
        if partition_stats['population'][old_district] > 0:
            old_total_pop = partition_stats['population'][old_district] + pixel_pop
            old_center_x = partition_stats['center_x'][old_district]
            old_center_y = partition_stats['center_y'][old_district]
            
            # Remove the pixel from old district center calculation
            # We use global i coordinate for center calculation
            partition_stats['center_x'][old_district] = (old_center_x * old_total_pop - j * pixel_pop) / partition_stats['population'][old_district]
            partition_stats['center_y'][old_district] = (old_center_y * old_total_pop - i * pixel_pop) / partition_stats['population'][old_district]
        
        # Add to new district center calculation
        new_total_pop = partition_stats['population'][new_district]
        if new_total_pop > 0:
            old_new_pop = new_total_pop - pixel_pop
            if old_new_pop > 0:
                old_center_x = partition_stats['center_x'][new_district]
                old_center_y = partition_stats['center_y'][new_district]
                
                partition_stats['center_x'][new_district] = (old_center_x * old_new_pop + j * pixel_pop) / new_total_pop
                partition_stats['center_y'][new_district] = (old_center_y * old_new_pop + i * pixel_pop) / new_total_pop
            else:
                partition_stats['center_x'][new_district] = j
                partition_stats['center_y'][new_district] = i
        
        # Update area
        partition_stats['area'][old_district] -= 1
        partition_stats['area'][new_district] += 1
        
        # Simplified perimeter update for multi-GPU
        # This is approximate but fast for partition-based processing
        perimeter_old = 0
        perimeter_new = 0
        
        # Check neighborhood
        for ni, nj in [(local_i+1, j), (local_i-1, j), (local_i, j+1), (local_i, j-1)]:
            if 0 <= ni < partition_state.shape[0] and 0 <= nj < partition_state.shape[1]:
                neighbor_district = int(partition_state[ni, nj, 0].item())
                if neighbor_district == old_district:
                    perimeter_old += 1
                elif neighbor_district == new_district:
                    perimeter_new -= 1
        
        partition_stats['perimeter'][old_district] += perimeter_old
        partition_stats['perimeter'][new_district] += perimeter_new
    
    def _score_partition(self, partition_map, partition_stats, partition_state):
        """Score a partition based on our metrics"""
        score = 0
        
        # Population equality score
        pop_std = cp.std(partition_stats['population'])
        pop_mean = cp.mean(partition_stats['population'])
        if pop_mean > 0:
            pop_score = (pop_std / pop_mean) ** 4
            score += self.weights['population_equality'] * pop_score
        
        # Compactness score
        compactness_scores = partition_stats['perimeter'] / cp.sqrt(partition_stats['area'])
        compactness_score = cp.mean(compactness_scores)
        score += self.weights['compactness'] * compactness_score
        
        # Election results score (simplified for partition)
        if self.weights['election_results'] > 0 and self.target_vote_margins is not None:
            vote_margins = []
            for district_id in range(self.num_districts):
                red = float(partition_stats['red_votes'][district_id])
                blue = float(partition_stats['blue_votes'][district_id])
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
            margins_error = cp.mean((cp.array(vote_margins) - cp.array(target_margins)) ** 2)
            score += self.weights['election_results'] * margins_error
        
        return score

    def update_phase(self, iteration, max_iterations):
        """Update the phase and adjust weights accordingly with improved phase transitions"""
        # Define phase transition points
        phase1_end = int(0.25 * max_iterations)  # 25% - Focus on population equality
        phase2_end = int(0.40 * max_iterations)  # 40% - Focus on vote distribution
        phase3_end = int(0.75 * max_iterations)  # 75% - Focus on compactness
        # Phase 4 - Final refinement (remaining 25%)
        
        # Determine current phase based on iteration
        if iteration == 0:
            # Phase 1: Focus almost exclusively on population equality
            self.phase = 1
            self.weights = {
                'population_equality': 3,  # Very high weight for population equality
                'compactness': 0,          # Low weight for compactness
                'center_distance': 0.5,    # Ignore center distance
                'election_results': 0      # Ignore election results initially
            }
            self.temperature = 1.0         # Start with max temperature to accept almost any change
            print(f"Iteration {iteration}: Phase 1 - Equalizing population")
            
        elif iteration == phase1_end:
            # Phase 2: Begin focusing on election results/vote distribution
            self.phase = 2
            self.weights = {
                'population_equality': 1,  # Still high but reduced
                'compactness': 2,          # Still low
                'center_distance': 2,      # Still ignored
                'election_results': 5      # Start optimizing for vote distribution
            }
            self.temperature = 0.7        # Still accepting many worse solutions
            print(f"Iteration {iteration}: Phase 2 - Optimizing vote distribution")
        
        elif iteration == phase2_end:
            # Phase 3: Focus on compactness while maintaining population equality and vote distribution
            self.phase = 3
            self.weights = {
                'population_equality': 1,  # Reduced but still important
                'compactness': 3,          # Increased focus on shape
                'center_distance': 3,      # Some focus on center distance
                'election_results': 2      # Decreased but still important
            }
            self.temperature = 0.3        # Less willing to accept worse solutions
            print(f"Iteration {iteration}: Phase 3 - Improving district compactness")
        
        elif iteration == phase3_end:
            # Phase 4: Final refinement
            self.phase = 4
            self.weights = {
                'population_equality': 1,  # Still important
                'compactness': 1,          # Very high focus on compactness
                'center_distance': 1,      # Higher focus on center distance
                'election_results': 2      # Still maintained but reduced
            }
            self.temperature = 0.1        # Only accept small increases in score
            print(f"Iteration {iteration}: Phase 4 - Final refinement")

    def set_target_vote_distribution(self, distribution_type, red_proportion=None):
        """
        Set the target vote distribution
        
        Parameters:
        - distribution_type: 'fair', 'red_gerrymander', 'blue_gerrymander', or 'incumbent'
        - red_proportion: Overall proportion of red votes (used for fair distribution)
        """
        if red_proportion is None:
            # Calculate from the current map
            if self.use_gpu:
                total_red = float(cp.sum(self.state_map_gpu[:,:,1]))
                total_blue = float(cp.sum(self.state_map_gpu[:,:,2]))
            else:
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

    def plot_districts(self, ax=None, show_stats=True):
        """Plot the current district map with a distinct background color"""
        # If using GPU, bring data to CPU for plotting
        if self.use_gpu:
            if self.using_dask:
                district_map_np = self.district_map.compute()
                valid_mask_np = self.valid_mask.compute()
            else:
                district_map_np = cp.asnumpy(self.district_map)
                valid_mask_np = cp.asnumpy(self.valid_mask)
        else:
            district_map_np = self.district_map
            valid_mask_np = self.valid_mask
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a masked array where invalid areas are masked
        masked_district_map = np.ma.masked_array(
            district_map_np, 
            mask=~valid_mask_np
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
            
            # Calculate total votes
            if self.use_gpu:
                if self.using_dask:
                    total_red = float(self.state_map_gpu[:,:,1].sum().compute())
                    total_blue = float(self.state_map_gpu[:,:,2].sum().compute())
                else:
                    total_red = float(cp.sum(self.state_map_gpu[:,:,1]))
                    total_blue = float(cp.sum(self.state_map_gpu[:,:,2]))
            else:
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
        if self.use_gpu:
            if self.using_dask:
                np.save(output_file, self.district_map.compute())
            else:
                np.save(output_file, cp.asnumpy(self.district_map))
        else:
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
    
    def cleanup(self):
        """Clean up resources when simulator is no longer needed"""
        # Close multiprocessing pool if used
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None
        
        # Close Dask client and cluster if used
        if self.use_gpu and self.num_gpus > 1 and self.using_dask:
            if hasattr(self, 'client') and self.client:
                self.client.close()
            if hasattr(self, 'cluster') and self.cluster:
                self.cluster.close()
            
        # Release CUDA contexts if used
        if self.use_gpu:
            for device in self.devices:
                device.use()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

                    
    def _process_single_pixel_move_partition(self, partition_map, partition_state, partition_mask, 
                                          partition_stats, temperature, phase, start_row, end_row):
        """Process a single pixel move on a partition"""
        # Get boundary pixels for this partition
        boundary_pixels = self._get_boundary_pixels_partition(partition_map, partition_mask)
        
        if len(boundary_pixels) == 0:
            return False
            
        # Randomly select a boundary pixel
        idx = np.random.randint(0, len(boundary_pixels))
        local_i, j = boundary_pixels[idx]
        # Convert to global coordinates for district tracking
        i = local_i + start_row
        
        old_district = int(partition_map[local_i, j].item())
        
        # Find neighboring districts within this partition
        neighboring_districts = set()
        
        for ni, nj in [(local_i+1, j), (local_i-1, j), (local_i, j+1), (local_i, j-1)]:
            if 0 <= ni < partition_map.shape[0] and 0 <= nj < partition_map.shape[1]:
                if partition_mask[ni, nj]:
                    neighboring_districts.add(int(partition_map[ni, nj].item()))
        
        neighboring_districts.discard(old_district)
        
        if not neighboring_districts:
            return False
            
        new_district = random.choice(list(neighboring_districts))
        
        # Check if this would break connectivity - simplified check for partition
        if self._will_break_district_partition(partition_map, local_i, j, old_district):
            return False
            
        # Score before change
        current_score = self._score_partition(partition_map, partition_stats, partition_state)
        
        # Make the change
        partition_map[local_i, j] = new_district
        self._update_stats_partition(partition_stats, partition_state, 
                                   local_i, j, old_district, new_district, start_row)
        
        # Score after change
        new_score = self._score_partition(partition_map, partition_stats, partition_state)
        
        # Decide whether to accept
        if new_score <= current_score:
            return True
        else:
            # Probabilistic acceptance based on temperature
            relative_diff = (new_score - current_score) / (current_score + 1e-10)
            accept_probability = cp.exp(-relative_diff / max(temperature, 0.001))
            
            if cp.random.random() < accept_probability:
                return True
            else:
                # Revert
                partition_map[local_i, j] = old_district
                self._update_stats_partition(partition_stats, partition_state,
                                          local_i, j, new_district, old_district, start_row)
                return False
    
    def _process_multi_pixel_move_partition(self, partition_map, partition_state, partition_mask, 
                                          partition_stats, temperature, phase, pixels_per_move,
                                          start_row, end_row):
        """Process multiple pixel moves at once on a partition"""
        # Get boundary pixels
        boundary_pixels = self._get_boundary_pixels_partition(partition_map, partition_mask)
        
        if len(boundary_pixels) == 0:
            return False
            
        # Choose random subset of boundary pixels
        if len(boundary_pixels) <= pixels_per_move:
            selected_pixels = boundary_pixels
        else:
            indices = np.random.choice(len(boundary_pixels), pixels_per_move, replace=False)
            selected_pixels = [boundary_pixels[i] for i in indices]
        
        # For each pixel, find potential moves
        moves = []
        
        for local_i, j in selected_pixels:
            # Convert to global coordinates
            i = local_i + start_row
            
            old_district = int(partition_map[local_i, j].item())
            
            # Find neighboring districts
            neighboring_districts = set()
            for ni, nj in [(local_i+1, j), (local_i-1, j), (local_i, j+1), (local_i, j-1)]:
                if 0 <= ni < partition_map.shape[0] and 0 <= nj < partition_map.shape[1]:
                    if partition_mask[ni, nj]:
                        neighboring_districts.add(int(partition_map[ni, nj].item()))
            
            neighboring_districts.discard(old_district)
            
            if neighboring_districts and not self._will_break_district_partition(
                    partition_map, local_i, j, old_district):
                new_district = random.choice(list(neighboring_districts))
                moves.append((local_i, j, old_district, new_district))
        
        if not moves:
            return False
            
        # Score before changes
        current_score = self._score_partition(partition_map, partition_stats, partition_state)
        
        # Make a backup of the current state
        original_map = cp.copy(partition_map)
        original_stats = {k: cp.copy(v) for k, v in partition_stats.items()}
        
        # Apply all moves
        for local_i, j, old_district, new_district in moves:
            partition_map[local_i, j] = new_district
            self._update_stats_partition(partition_stats, partition_state,
                                       local_i, j, old_district, new_district, start_row)
        
        # Score after changes
        new_score = self._score_partition(partition_map, partition_stats, partition_state)
        
        # Decide whether to accept all changes
        if new_score <= current_score:
            return True
        else:
            # For early phases, be more lenient
            if phase == 1:
                accept_probability = 0.8
            else:
                # Use temperature
                relative_diff = (new_score - current_score) / (current_score + 1e-10)
                accept_probability = cp.exp(-relative_diff / max(temperature, 0.001))
            
            if cp.random.random() < accept_probability:
                return True
            else:
                # Revert all changes
                partition_map[:] = original_map
                for k in partition_stats:
                    partition_stats[k][:] = original_stats[k]
                return False
    
    def _get_boundary_pixels_partition(self, partition_map, partition_mask):
        """Get boundary pixels within a partition"""
        # Pre-allocate arrays for the four neighbor directions
        up_shifted = cp.pad(partition_map[:-1, :], ((1, 0), (0, 0)), mode='constant', constant_values=-1)
        down_shifted = cp.pad(partition_map[1:, :], ((0, 1), (0, 0)), mode='constant', constant_values=-1)
        left_shifted = cp.pad(partition_map[:, :-1], ((0, 0), (1, 0)), mode='constant', constant_values=-1)
        right_shifted = cp.pad(partition_map[:, 1:], ((0, 0), (0, 1)), mode='constant', constant_values=-1)
        
        # A pixel is on a boundary if any of its neighbors are in a different district
        is_boundary = ((up_shifted != partition_map) | 
                     (down_shifted != partition_map) | 
                     (left_shifted != partition_map) | 
                     (right_shifted != partition_map)) & partition_mask
        
        # Get coordinates of boundary pixels
        boundary_pixels = cp.argwhere(is_boundary).get()
        
        return boundary_pixels
    
    def run_simulation_multi_gpu(self, num_iterations=100000, batch_size=1000, pixels_per_move=30):
        """
        Run simulation optimized for multiple GPUs using partition-based processing
        
        Parameters:
        - num_iterations: Total number of iterations to run
        - batch_size: Number of iterations per batch
        - pixels_per_move: Number of pixels to change in each move (for multi-pixel moves)
        """
        if not self.use_gpu or self.num_gpus <= 1:
            print("Multi-GPU simulation requires multiple GPUs. Falling back to single GPU/CPU mode.")
            return self.run_simulation(num_iterations, batch_size, True, pixels_per_move)
        
        print(f"Running multi-GPU simulation on {self.num_gpus} GPUs")
        
        # Initialize the phase
        self.update_phase(0, num_iterations)
        
        iterations_completed = 0
        progress_bar = tqdm(total=num_iterations)
        
        while iterations_completed < num_iterations:
            # Calculate iterations for this batch
            current_batch_size = min(batch_size, num_iterations - iterations_completed)
            
            # Distribute work across GPUs
            iterations_per_gpu = max(1, current_batch_size // self.num_gpus)
            
            if self.using_dask:
                # Use Dask for distributed processing
                futures = []
                
                for gpu_id in range(self.num_gpus):
                    # Calculate this GPU's workload
                    gpu_iterations = iterations_per_gpu
                    if gpu_id == self.num_gpus - 1:
                        # Last GPU gets any remainder
                        gpu_iterations += current_batch_size % self.num_gpus
                    
                    # Skip if no work
                    if gpu_iterations <= 0:
                        continue
                    
                    # Determine region for this GPU
                    start_row, end_row = self.partition_boundaries[gpu_id]
                    
                    # Submit task to GPU
                    futures.append(self.client.submit(
                        self._process_batch_gpu_partition,
                        gpu_id, gpu_iterations, self.phase, self.temperature,
                        pixels_per_move if self.phase <= 2 else 1,
                        start_row, end_row,
                        resources={'GPU': 1}
                    ))
                
                # Wait for all tasks to complete
                results = self.client.gather(futures)
                
                # Process results
                district_maps = []
                total_accepted = 0
                best_score = float('inf')
                best_idx = -1
                
                for i, (partition_map, accepted, score) in enumerate(results):
                    total_accepted += accepted
                    if score < best_score:
                        best_score = score
                        best_idx = i
                
                # If we found a better partition, update the full map
                if best_idx >= 0:
                    start_row, end_row = self.partition_boundaries[best_idx]
                    best_partition = results[best_idx][0]
                    
                    # Convert to dask array and update
                    partition_dask = da.from_array(best_partition, 
                                                chunks=(end_row-start_row, self.width))
                    self.district_map[start_row:end_row] = partition_dask
            else:
                # Manual multi-GPU management
                # Process each partition on its GPU
                results = []
                for gpu_id in range(self.num_gpus):
                    # Calculate this GPU's workload
                    gpu_iterations = iterations_per_gpu
                    if gpu_id == self.num_gpus - 1:
                        # Last GPU gets any remainder
                        gpu_iterations += current_batch_size % self.num_gpus
                    
                    # Skip if no work
                    if gpu_iterations <= 0:
                        continue
                    
                    # Determine region for this GPU
                    start_row, end_row = self.partition_boundaries[gpu_id]
                    
                    # Process partition
                    with cp.cuda.Device(gpu_id):
                        result = self._process_batch_gpu_partition(
                            gpu_id, gpu_iterations, self.phase, self.temperature,
                            pixels_per_move if self.phase <= 2 else 1,
                            start_row, end_row)
                        results.append(result)
                
                # Process results
                total_accepted = 0
                best_score = float('inf')
                best_idx = -1
                
                for i, (partition_map, accepted, score) in enumerate(results):
                    total_accepted += accepted
                    if score < best_score:
                        best_score = score
                        best_idx = i
                
                # If we found a better partition, update the full map
                if best_idx >= 0:
                    start_row, end_row = self.partition_boundaries[best_idx]
                    best_partition = results[best_idx][0]
                    
                    # Update the district map
                    with cp.cuda.Device(0):  # Use device 0 for updates
                        self.district_map[start_row:end_row] = cp.asarray(best_partition)
            
            # Update iteration count
            iterations_completed += current_batch_size
            
            # Update phase and temperature
            self.update_phase(iterations_completed, num_iterations)
            self.temperature *= self.cooling_rate
            
            # Update progress bar
            progress_bar.update(current_batch_size)
            
            # Recalculate statistics periodically
            if iterations_completed % 1000 == 0:
                self.calculate_all_district_stats()
                
                # Display progress info
                pop_mean = float(cp.mean(self.district_stats_gpu['population']))
                pop_max = float(cp.max(self.district_stats_gpu['population']))
                pop_min = float(cp.min(self.district_stats_gpu['population']))
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
        print("Multi-GPU simulation complete!")
        
        # Calculate final statistics
        self.calculate_all_district_stats()
    
    def _process_batch_gpu_partition(self, gpu_id, batch_size, phase, temperature, pixels_per_move, start_row, end_row):
        """
        Process a batch of iterations on a specific GPU partition
        
        Returns:
        - Tuple of (updated partition map, accepted count, best score)
        """
        with cp.cuda.Device(gpu_id):
            # Get the partition for this GPU
            if self.using_dask:
                partition_map = cp.asarray(self.district_map[start_row:end_row].compute())
                partition_state = cp.asarray(self.state_map_gpu[start_row:end_row].compute())
                partition_mask = cp.asarray(self.valid_mask[start_row:end_row].compute())
            else:
                partition_map = cp.asarray(self.district_map[start_row:end_row].get())
                partition_state = cp.asarray(self.state_map_gpu[start_row:end_row].get())
                partition_mask = cp.asarray(self.valid_mask[start_row:end_row].get())
            
            # Copy district stats for this partition
            partition_stats = {k: cp.copy(v) for k, v in self.district_stats_gpu.items()}
            
            # Track best score and corresponding map
            best_score = float('inf')
            best_map = cp.copy(partition_map)
            accepted_count = 0
            
            # Process the batch
            for _ in range(batch_size):
                if phase <= 2 and pixels_per_move > 1:
                    # Multi-pixel mode for early phases
                    accepted = self._process_multi_pixel_move_partition(
                        partition_map, partition_state, partition_mask, 
                        partition_stats, temperature, phase, pixels_per_move,
                        start_row, end_row)
                    if accepted:
                        accepted_count += 1
                else:
                    # Single pixel mode
                    accepted = self._process_single_pixel_move_partition(
                        partition_map, partition_state, partition_mask,
                        partition_stats, temperature, phase,
                        start_row, end_row)
                    if accepted:
                        accepted_count += 1
                
                # Check if this is the best map so far
                current_score = self._score_partition(partition_map, partition_stats, partition_state)
                if current_score < best_score:
                    best_score = current_score
                    best_map = cp.copy(partition_map)
            
            # Return the results
            return best_map.get(), accepted_count, float(best_score)   
    def _calculate_perimeter_partition_gpu(self, partition_map, district_id):
        """Calculate perimeter of a district in a partition using GPU"""
        # Define CUDA kernel for perimeter calculation
        @cuda.jit
        def perimeter_kernel(district_map, district_id, height, width, result):
            # Get thread position
            i, j = cuda.grid(2)
            
            # Check if in bounds and in the district
            if i < height and j < width and district_map[i, j] == district_id:
                perimeter_count = 0
                
                # Check neighbors
                for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                    if (0 <= ni < height and 0 <= nj < width):
                        if district_map[ni, nj] != district_id:
                            perimeter_count += 1
                    else:
                        # Edge of the partition - we count this as potential boundary
                        perimeter_count += 1
                
                # Use atomic add to avoid race conditions
                cuda.atomic.add(result, 0, perimeter_count)
        
        # Setup for CUDA kernel
        height, width = partition_map.shape
        
        # Set up grid and block dimensions
        threads_per_block = (16, 16)
        blocks_per_grid_x = (height + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (width + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Prepare arrays for CUDA
        d_district_map = cuda.to_device(cp.asnumpy(partition_map))
        d_result = cuda.to_device(np.zeros(1, dtype=np.int32))
        
        # Launch kernel
        perimeter_kernel[blocks_per_grid, threads_per_block](
            d_district_map, district_id, height, width, d_result)
        
        # Get result
        result = d_result.copy_to_host()
        return result[0]
    
    def _calculate_perimeter_gpu(self, district_id, gpu_id=0, start_row=0, end_row=None):
        """Calculate perimeter of a district using GPU"""
        end_row = end_row or self.height
        
        # Define CUDA kernel for perimeter calculation
        @cuda.jit
        def perimeter_kernel(district_map, district_id, height, width, result):
            # Get thread position
            i, j = cuda.grid(2)
            
            # Check if in bounds and in the district
            if i < height and j < width and district_map[i, j] == district_id:
                perimeter_count = 0
                
                # Check neighbors
                for ni, nj in [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]:
                    if (0 <= ni < height and 0 <= nj < width):
                        if district_map[ni, nj] != district_id:
                            perimeter_count += 1
                    else:
                        # Edge of the grid
                        perimeter_count += 1
                
                # Use atomic add to avoid race conditions
                cuda.atomic.add(result, 0, perimeter_count)
        
        # Context manager for GPU device
        with cp.cuda.Device(gpu_id):
            # Get the right partition
            if self.num_gpus > 1:
                if self.using_dask:
                    district_map_np = cp.asnumpy(self.district_map[start_row:end_row].compute())
                else:
                    district_map_np = cp.asnumpy(self.district_map[start_row:end_row])
            else:
                district_map_np = cp.asnumpy(self.district_map)
            
            # Set up grid and block dimensions
            height, width = district_map_np.shape
            threads_per_block = (16, 16)
            blocks_per_grid_x = (height + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (width + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            
            # Prepare arrays for CUDA
            d_district_map = cuda.to_device(district_map_np)
            d_result = cuda.to_device(np.zeros(1, dtype=np.int32))
            
            # Launch kernel
            perimeter_kernel[blocks_per_grid, threads_per_block](
                d_district_map, district_id, height, width, d_result)
            
            # Get result
            result = d_result.copy_to_host()
            return result[0]
    
    @staticmethod
    def calc_perimeter_chunk(params):
        """
        Calculate perimeter for a chunk of pixels - used for CPU processing
        
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
    def _calculate_partition_stats(self, gpu_id, start_row, end_row, num_districts):
        """Calculate statistics for a partition of the map on a specific GPU"""
        # Set device context
        with cp.cuda.Device(gpu_id):
            # Get the partition for this GPU
            if self.using_dask:
                partition_map = cp.asarray(self.district_map[start_row:end_row].compute())
                partition_state = cp.asarray(self.state_map_gpu[start_row:end_row].compute())
                valid_partition = cp.asarray(self.valid_mask[start_row:end_row].compute())
            else:
                partition_map = self.district_map[start_row:end_row]
                partition_state = self.state_map_gpu[start_row:end_row]
                valid_partition = self.valid_mask[start_row:end_row]
            
            # Initialize stats for this partition
            stats = {
                'population': cp.zeros(num_districts),
                'red_votes': cp.zeros(num_districts),
                'blue_votes': cp.zeros(num_districts),
                'center_x': cp.zeros(num_districts),
                'center_y': cp.zeros(num_districts),
                'perimeter': cp.zeros(num_districts),
                'area': cp.zeros(num_districts)
            }
            
            # Find unique districts in this partition
            district_ids = cp.unique(partition_map[valid_partition])
            
            for district_id in district_ids.get():
                mask = (partition_map == district_id)
                
                # Count population and votes using vectorized operations
                stats['population'][district_id] = cp.sum(partition_state[:,:,0] * mask)
                stats['red_votes'][district_id] = cp.sum(partition_state[:,:,1] * mask)
                stats['blue_votes'][district_id] = cp.sum(partition_state[:,:,2] * mask)
                
                # Calculate center of population
                if stats['population'][district_id] > 0:
                    pop_indices = cp.argwhere(mask & (partition_state[:,:,0] > 0))
                    if len(pop_indices) > 0:
                        pop_weights = cp.array([partition_state[i, j, 0] for i, j in pop_indices])
                        
                        # Adjust row indices to global coordinates
                        global_indices = pop_indices.copy()
                        global_indices[:, 0] += start_row
                        
                        # Store weighted sum for later normalization
                        stats['center_y'][district_id] = cp.sum(global_indices[:, 0] * pop_weights)
                        stats['center_x'][district_id] = cp.sum(global_indices[:, 1] * pop_weights)
                
                # Area is simple
                stats['area'][district_id] = cp.sum(mask)
                
                # Calculate perimeter using GPU kernel
                perimeter = self._calculate_perimeter_partition_gpu(partition_map, district_id)
                stats['perimeter'][district_id] = perimeter
            
            # Return the stats dictionary for this partition
            return {k: v.get() for k, v in stats.items()}