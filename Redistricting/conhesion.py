"""
District cohesion enhancement for the GerrymanderSimulator.

This module provides functions to enhance district cohesion by targeting
isolated pixels or pixels surrounded by different districts.
"""

import numpy as np
from tqdm import tqdm
import random

def get_neighbor_district_count(self, pixel_i, pixel_j):
    """
    Count how many neighboring pixels belong to different districts
    
    Parameters:
    - pixel_i, pixel_j: Coordinates of the pixel to check
    
    Returns:
    - Number of neighboring pixels belonging to different districts, 
      and a dictionary of neighboring district counts
    """
    current_district = self.district_map[pixel_i, pixel_j]
    neighbor_districts = {}
    different_count = 0
    
    # Check 8-way neighbors (including diagonals)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue  # Skip the center pixel
                
            ni, nj = pixel_i + di, pixel_j + dj
            
            # Check bounds
            if 0 <= ni < self.height and 0 <= nj < self.width and self.valid_mask[ni, nj]:
                neighbor_district = self.district_map[ni, nj]
                
                if neighbor_district != current_district:
                    different_count += 1
                    
                    # Count occurrences of each district
                    if neighbor_district in neighbor_districts:
                        neighbor_districts[neighbor_district] += 1
                    else:
                        neighbor_districts[neighbor_district] = 1
    
    return different_count, neighbor_districts

def identify_isolated_pixels(self, threshold=5):
    """
    Find pixels that are surrounded by many pixels from different districts
    
    Parameters:
    - threshold: Minimum number of different neighbors to consider a pixel isolated
    
    Returns:
    - List of (i, j, district_id, most_common_neighbor_district) for isolated pixels
    """
    isolated_pixels = []
    
    # Get all boundary pixels
    boundary_pixels = self.get_boundary_pixels()
    
    # Check each boundary pixel
    for pixel_i, pixel_j in boundary_pixels:
        different_count, neighbor_districts = self.get_neighbor_district_count(pixel_i, pixel_j)
        
        if different_count >= threshold:
            current_district = self.district_map[pixel_i, pixel_j]
            
            # Find the most common neighboring district
            most_common_district = max(neighbor_districts.items(), key=lambda x: x[1])[0]
            
            isolated_pixels.append((pixel_i, pixel_j, current_district, most_common_district))
    
    return isolated_pixels

def prioritize_moves_by_isolation(self, num_iterations=1000, isolation_threshold=4, 
                                  batch_size=100, weighting_factor=5.0):
    """
    Run improvement iterations that prioritize moves based on pixel isolation
    
    Parameters:
    - num_iterations: Number of iterations to run
    - isolation_threshold: Minimum number of different neighbors to consider a pixel isolated
    - batch_size: Batch size for iterations
    - weighting_factor: How much more likely isolated pixels are to be selected
    
    Returns:
    - Number of isolated pixels fixed
    """
    print(f"Running cohesion enhancement phase for {num_iterations} iterations...")
    
    # Increase compactness weight temporarily
    original_weights = self.weights.copy()
    self.weights['compactness'] *= 2.0
    
    iterations_completed = 0
    progress_bar = tqdm(total=num_iterations)
    isolated_pixels_fixed = 0
    
    while iterations_completed < num_iterations:
        current_batch_size = min(batch_size, num_iterations - iterations_completed)
        
        # Find isolated pixels
        isolated_pixels = self.identify_isolated_pixels(threshold=isolation_threshold)
        
        if not isolated_pixels:
            print("No isolated pixels found, switching to regular optimization")
            break
            
        # Process batch
        for _ in range(current_batch_size):
            # With some probability, pick an isolated pixel instead of a random boundary pixel
            if random.random() < 0.8:  # 80% chance to prioritize isolated pixels
                if isolated_pixels:
                    # Pick a random isolated pixel
                    pixel_i, pixel_j, old_district, new_district = random.choice(isolated_pixels)
                    
                    # Check if this would break the district
                    if not self.will_break_district(pixel_i, pixel_j, old_district):
                        # Make the change and update stats
                        self.district_map[pixel_i, pixel_j] = new_district
                        self.update_district_stats(pixel_i, pixel_j, old_district, new_district)
                        isolated_pixels_fixed += 1
                    else:
                        # Fall back to regular iteration
                        self.run_iteration()
                else:
                    self.run_iteration()
            else:
                # Run regular iteration
                self.run_iteration()
        
        iterations_completed += current_batch_size
        progress_bar.update(current_batch_size)
    
    progress_bar.close()
    
    # Restore original weights
    self.weights = original_weights
    
    print(f"Cohesion enhancement completed. Fixed {isolated_pixels_fixed} isolated pixels.")
    return isolated_pixels_fixed

def run_anti_fragmentation_phase(self):
    """
    Run a multi-step process to reduce fragmentation and increase district cohesion
    
    Parameters: None
    
    Returns:
    - True if improvements were made
    """
    print("Running anti-fragmentation process...")
    
    # Step 1: Store the original state
    original_weights = self.weights.copy()
    original_temperature = self.temperature
    
    # Step 2: Set weights to heavily favor compactness and center distance
    self.weights = {
        'population_equality': 1.0,  # Maintain population balance
        'compactness': 10.0,        # Extremely high weight for compactness
        'center_distance': 5.0,     # Strongly favor cohesive districts
        'election_results': 0.5     # Reduce partisan emphasis during this phase
    }
    
    # Step 3: Increase temperature to allow more movement
    self.temperature = 0.5
    
    # Step 4: Run isolated pixel prioritization
    isolated_fixed = self.prioritize_moves_by_isolation(num_iterations=5000, 
                                                      isolation_threshold=4,
                                                      batch_size=100)
    
    # Step 5: Run additional cohesion improvement with modified weights
    self.temperature = 0.3  # Slightly lower temperature
    
    accepted = 0
    if hasattr(self, 'run_batch_parallel') and self.num_cpus > 1:
        accepted = self.run_batch_parallel(batch_size=5000)
    else:
        for _ in range(5000):
            if self.run_iteration():
                accepted += 1
    
    # Step 6: Focus specifically on "island" pixels (completely surrounded)
    isolated_fixed_extreme = self.prioritize_moves_by_isolation(num_iterations=1000, 
                                                             isolation_threshold=6,
                                                             batch_size=50)
    
    # Step 7: Restore original weights and temperature
    self.weights = original_weights
    self.temperature = original_temperature
    
    print(f"Anti-fragmentation process completed: fixed {isolated_fixed + isolated_fixed_extreme} isolated pixels")
    return True

def add_cohesion_methods(simulator_class):
    """
    Add district cohesion enhancement methods to the GerrymanderSimulator class
    
    Parameters:
    - simulator_class: The GerrymanderSimulator class
    
    Returns:
    - Enhanced simulator class
    """
    simulator_class.get_neighbor_district_count = get_neighbor_district_count
    simulator_class.identify_isolated_pixels = identify_isolated_pixels
    simulator_class.prioritize_moves_by_isolation = prioritize_moves_by_isolation
    simulator_class.run_anti_fragmentation_phase = run_anti_fragmentation_phase
    
    # Also need to modify the run_simulation method to incorporate an anti-fragmentation phase
    original_run_simulation = simulator_class.run_simulation
    
    def enhanced_run_simulation(self, num_iterations=100000, batch_size=1000, 
                              use_parallel=True, pixels_per_move=20,
                              anti_fragmentation=True, **kwargs):
        """
        Enhanced run_simulation that includes an anti-fragmentation phase
        
        Parameters:
        - Same as original run_simulation
        - anti_fragmentation: Whether to run the anti-fragmentation phase
        
        Returns:
        - Results from original run_simulation
        """
        # First run the original simulation
        result = original_run_simulation(self, num_iterations, batch_size, 
                                       use_parallel, pixels_per_move, **kwargs)
        
        # Run anti-fragmentation phase if requested
        if anti_fragmentation:
            print("\nStarting anti-fragmentation phase to make districts more cohesive...")
            self.run_anti_fragmentation_phase()
            
            # Recalculate stats
            self.calculate_all_district_stats()
        
        return result
    
    # Replace the run_simulation method
    simulator_class.run_simulation = enhanced_run_simulation
    
    return simulator_class

def optimize_boundary_moves(self):
    """
    Enhanced boundary pixel selection that favors moves that lead to more compact districts
    
    This modifies the run_iteration method to prefer boundary moves that:
    1. Reduce the isolation of pixels
    2. Lead to more compact districts
    
    Returns:
    - True if changes were made
    """
    # Get all boundary pixels
    boundary_pixels = self.get_boundary_pixels()
    
    if len(boundary_pixels) == 0:
        return False
    
    # Score each boundary pixel based on its isolation level
    scored_moves = []
    
    for pixel_i, pixel_j in boundary_pixels:
        old_district = self.district_map[pixel_i, pixel_j]
        
        # Get isolation score (higher means more isolated)
        isolation_score, neighbor_districts = self.get_neighbor_district_count(pixel_i, pixel_j)
        
        # Skip if no different neighbors
        if not neighbor_districts:
            continue
            
        # Check neighboring districts as potential targets
        for new_district, count in neighbor_districts.items():
            # Skip if this would break the district
            if self.will_break_district(pixel_i, pixel_j, old_district):
                continue
                
            # Calculate a score for this move (higher is better)
            # Count of neighbors of the same new district (higher is better)
            connectedness_score = count
            
            # Overall score is isolation + connectedness
            move_score = isolation_score + connectedness_score
            
            scored_moves.append((pixel_i, pixel_j, old_district, new_district, move_score))
    
    # If no valid moves, try regular iteration
    if not scored_moves:
        return self.run_iteration()
        
    # Sort moves by score (higher first)
    scored_moves.sort(key=lambda x: x[4], reverse=True)
    
    # Either pick the top move or sample with probability proportional to score
    if random.random() < 0.7:  # 70% chance to pick the top move
        pixel_i, pixel_j, old_district, new_district, _ = scored_moves[0]
    else:
        # Sample with probability proportional to score
        total_score = sum(move[4] for move in scored_moves)
        r = random.uniform(0, total_score)
        current = 0
        for pixel_i, pixel_j, old_district, new_district, score in scored_moves:
            current += score
            if current >= r:
                break
    
    # Make the change
    self.district_map[pixel_i, pixel_j] = new_district
    self.update_district_stats(pixel_i, pixel_j, old_district, new_district)
    
    return True