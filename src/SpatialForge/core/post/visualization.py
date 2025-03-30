"""
Enhanced visualization capabilities for the GerrymanderSimulator.

This module provides functions for creating heatmaps, tracking district statistics,
and comparing different gerrymandering scenarios.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import pandas as pd

def create_district_change_heatmap(self, num_iterations=10000, batch_size=1000, 
                                  sample_interval=100, output_file="district_changes.png"):
    """
    Run the simulation and track district changes for each pixel
    
    Parameters:
    - num_iterations: Total number of iterations to run
    - batch_size: Number of iterations per batch
    - sample_interval: Sample district map every N iterations
    - output_file: Output image file
    
    Returns:
    - The output filename
    """
    print(f"Running simulation and tracking district changes for {num_iterations} iterations...")
    start_time = time.time()
    
    # Initialize a map to track how many times each pixel changes districts
    change_count = np.zeros((self.height, self.width), dtype=np.int32)
    
    # Store the initial district map
    prev_district_map = self.district_map.copy()
    
    # Run the simulation and track changes
    iterations_completed = 0
    progress_bar = tqdm(total=num_iterations)
    
    while iterations_completed < num_iterations:
        # Determine current batch size
        current_batch_size = min(batch_size, num_iterations - iterations_completed)
        
        # Run a batch of iterations
        if hasattr(self, 'run_batch_parallel') and self.num_cpus > 1:
            accepted = self.run_batch_parallel(batch_size=current_batch_size)
        else:
            # Fall back to single-threaded processing
            for _ in range(current_batch_size):
                self.run_iteration()
        
        # Update iteration count
        iterations_completed += current_batch_size
        
        # Cool the temperature
        self.temperature *= self.cooling_rate ** current_batch_size
        
        # Update phase
        self.update_phase(iterations_completed, num_iterations)
        
        # Sample changes if it's time
        if iterations_completed % sample_interval == 0 or iterations_completed == num_iterations:
            # Count pixels that changed districts
            change_count += (self.district_map != prev_district_map) & self.valid_mask
            
            # Update previous map
            prev_district_map = self.district_map.copy()
        
        # Update progress bar
        progress_bar.update(current_batch_size)
    
    progress_bar.close()
    
    # Create a heatmap visualization
    plt.figure(figsize=(12, 10))
    
    # Create a custom colormap - white for no changes, increasing red intensity for more changes
    colors = [(1, 1, 1), (1, 0, 0)]  # White to red
    cmap_name = 'change_intensity'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    # Apply the mask to the change count
    masked_change_count = np.ma.masked_array(change_count, mask=~self.valid_mask)
    
    # Plot the heatmap
    plt.imshow(masked_change_count, cmap=cm)
    plt.colorbar(label='Number of district changes')
    plt.title('District Boundary Changes Heatmap')
    plt.axis('off')
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    end_time = time.time()
    print(f"Change heatmap created in {end_time - start_time:.2f} seconds")
    print(f"Heatmap saved to {output_file}")
    
    return output_file

def track_district_stats_over_time(self, num_iterations=10000, batch_size=1000, 
                                  sample_interval=500, output_file="district_stats_evolution.csv"):
    """
    Run the simulation and track district statistics over time
    
    Parameters:
    - num_iterations: Total number of iterations to run
    - batch_size: Number of iterations per batch  
    - sample_interval: Sample district stats every N iterations
    - output_file: Output CSV file
    
    Returns:
    - The output filename and a pandas DataFrame with the stats
    """
    print(f"Running simulation and tracking district stats for {num_iterations} iterations...")
    start_time = time.time()
    
    # Initialize a list to store the stats at each sample point
    stats_history = []
    
    # Run the simulation and track stats
    iterations_completed = 0
    progress_bar = tqdm(total=num_iterations)
    
    while iterations_completed < num_iterations:
        # Determine current batch size
        current_batch_size = min(batch_size, num_iterations - iterations_completed)
        
        # Run a batch of iterations
        if hasattr(self, 'run_batch_parallel') and self.num_cpus > 1:
            accepted = self.run_batch_parallel(batch_size=current_batch_size)
        else:
            # Fall back to single-threaded processing
            for _ in range(current_batch_size):
                self.run_iteration()
        
        # Update iteration count
        iterations_completed += current_batch_size
        
        # Cool the temperature
        self.temperature *= self.cooling_rate ** current_batch_size
        
        # Update phase
        self.update_phase(iterations_completed, num_iterations)
        
        # Sample stats if it's time
        if iterations_completed % sample_interval == 0 or iterations_completed == num_iterations:
            # Calculate current stats
            self.calculate_all_district_stats()
            
            # Create a record for this iteration
            stats_record = {
                'iteration': iterations_completed,
                'temperature': self.temperature,
                'phase': self.phase,
                'score': self.score_map()
            }
            
            # Add district-specific stats
            for district_id in range(self.num_districts):
                stats_record[f'district_{district_id}_population'] = self.district_stats['population'][district_id]
                stats_record[f'district_{district_id}_red_votes'] = self.district_stats['red_votes'][district_id]
                stats_record[f'district_{district_id}_blue_votes'] = self.district_stats['blue_votes'][district_id]
                
                # Calculate vote margin
                red = self.district_stats['red_votes'][district_id]
                blue = self.district_stats['blue_votes'][district_id]
                total = red + blue
                if total > 0:
                    margin = red / total
                else:
                    margin = 0.5
                stats_record[f'district_{district_id}_red_margin'] = margin
                
                # Calculate compactness
                perimeter = self.district_stats['perimeter'][district_id]
                area = self.district_stats['area'][district_id]
                if area > 0:
                    compactness = perimeter / np.sqrt(area)
                else:
                    compactness = 0
                stats_record[f'district_{district_id}_compactness'] = compactness
            
            # Add overall election results
            red_districts = sum(1 for d in range(self.num_districts) 
                               if self.district_stats['red_votes'][d] > self.district_stats['blue_votes'][d])
            blue_districts = self.num_districts - red_districts
            
            stats_record['red_districts'] = red_districts
            stats_record['blue_districts'] = blue_districts
            
            # Calculate overall vote shares
            total_red = sum(self.district_stats['red_votes'])
            total_blue = sum(self.district_stats['blue_votes'])
            total_votes = total_red + total_blue
            
            if total_votes > 0:
                stats_record['overall_red_vote_share'] = total_red / total_votes
                stats_record['overall_blue_vote_share'] = total_blue / total_votes
            else:
                stats_record['overall_red_vote_share'] = 0.5
                stats_record['overall_blue_vote_share'] = 0.5
            
            # Calculate efficiency gap (a gerrymandering metric)
            red_wasted = 0
            blue_wasted = 0
            
            for district_id in range(self.num_districts):
                red = self.district_stats['red_votes'][district_id]
                blue = self.district_stats['blue_votes'][district_id]
                total = red + blue
                
                if total > 0:
                    if red > blue:  # Red wins
                        threshold = total / 2 + 1
                        red_wasted += red - threshold
                        blue_wasted += blue
                    else:  # Blue wins
                        threshold = total / 2 + 1
                        red_wasted += red
                        blue_wasted += blue - threshold
            
            if total_votes > 0:
                stats_record['efficiency_gap'] = (red_wasted - blue_wasted) / total_votes
            else:
                stats_record['efficiency_gap'] = 0
            
            # Add to history
            stats_history.append(stats_record)
        
        # Update progress bar
        progress_bar.update(current_batch_size)
    
    progress_bar.close()
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_history)
    
    # Save to CSV
    stats_df.to_csv(output_file, index=False)
    
    end_time = time.time()
    print(f"District stats tracking completed in {end_time - start_time:.2f} seconds")
    print(f"Stats saved to {output_file}")
    
    return output_file, stats_df

def plot_metrics_evolution(stats_df, simulator, output_file="metrics_evolution.png"):
    """
    Plot the evolution of key metrics from the stats
    
    Parameters:
    - stats_df: DataFrame with stats history
    - simulator: GerrymanderSimulator instance
    - output_file: Output filename for the plot
    
    Returns:
    - The output filename
    """
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(stats_df['iteration'], stats_df['score'])
    plt.title('Optimization Score')
    plt.xlabel('Iteration')
    plt.ylabel('Score (lower is better)')
    plt.yscale('log')
    
    plt.subplot(2, 2, 2)
    plt.plot(stats_df['iteration'], stats_df['red_districts'], 'r-', label='Red Districts')
    plt.plot(stats_df['iteration'], stats_df['blue_districts'], 'b-', label='Blue Districts')
    plt.axhline(y=stats_df['overall_red_vote_share'].iloc[-1] * simulator.num_districts, color='r', linestyle='--', 
               label='Proportional Red Seats')
    plt.axhline(y=stats_df['overall_blue_vote_share'].iloc[-1] * simulator.num_districts, color='b', linestyle='--',
               label='Proportional Blue Seats')
    plt.title('Partisan Composition of Districts')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Districts')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(stats_df['iteration'], stats_df['efficiency_gap'])
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Efficiency Gap')
    plt.xlabel('Iteration')
    plt.ylabel('Efficiency Gap (+ favors Red)')
    
    plt.subplot(2, 2, 4)
    plt.plot(stats_df['iteration'], stats_df['temperature'])
    plt.title('Annealing Temperature')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return output_file

def compare_scenarios_metrics(all_stats, num_districts, output_file="metrics_comparison.png"):
    """
    Create a comparison of key metrics across different scenarios
    
    Parameters:
    - all_stats: Dictionary of DataFrames with stats for each scenario
    - num_districts: Number of districts in the simulation
    - output_file: Output filename for the plot
    
    Returns:
    - The output filename
    """
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for scenario, df in all_stats.items():
        plt.plot(df['iteration'], df['score'], label=scenario)
    plt.title('Optimization Score')
    plt.xlabel('Iteration')
    plt.ylabel('Score (lower is better)')
    plt.yscale('log')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for scenario, df in all_stats.items():
        # Calculate seat proportion vs vote proportion
        red_seat_prop = df['red_districts'] / num_districts
        red_vote_prop = df['overall_red_vote_share']
        plt.plot(df['iteration'], red_seat_prop - red_vote_prop, label=scenario)
    
    plt.axhline(y=0, color='k', linestyle='--', label='Proportional')
    plt.title('Partisan Bias (Red Seat % - Red Vote %)')
    plt.xlabel('Iteration')
    plt.ylabel('Bias')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for scenario, df in all_stats.items():
        plt.plot(df['iteration'], df['efficiency_gap'], label=scenario)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Efficiency Gap')
    plt.xlabel('Iteration')
    plt.ylabel('Efficiency Gap (+ favors Red)')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    
    # Calculate and plot compactness
    compactness_by_scenario = {}
    for scenario, df in all_stats.items():
        # Get the average compactness across all districts for the final iteration
        compactness_cols = [col for col in df.columns if 'compactness' in col]
        compactness_by_scenario[scenario] = df[compactness_cols].iloc[-1].mean()
    
    scenarios = list(compactness_by_scenario.keys())
    plt.bar(scenarios, [compactness_by_scenario[s] for s in scenarios])
    plt.title('Final Compactness by Scenario')
    plt.ylabel('Average Compactness (lower is better)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return output_file

def create_comparison_visualization(scenario_results, output_file="scenario_comparison.png"):
    """
    Create a visual comparison of different scenarios
    
    Parameters:
    - scenario_results: Dictionary of results for each scenario
    - output_file: Output filename
    
    Returns:
    - The output filename
    """
    plt.figure(figsize=(20, 15))
    
    scenarios = list(scenario_results.keys())
    gs = gridspec.GridSpec(2, len(scenarios))
    
    # Plot the district maps
    for i, scenario in enumerate(scenarios):
        ax = plt.subplot(gs[0, i])
        img = plt.imread(scenario_results[scenario]['final_districts'])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{scenario.replace('_', ' ').title()}")
    
    # Plot the heatmaps
    for i, scenario in enumerate(scenarios):
        ax = plt.subplot(gs[1, i])
        img = plt.imread(scenario_results[scenario]['heatmap'])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"District Changes - {scenario.replace('_', ' ').title()}")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    return output_file

def add_visualization_methods(GerrymanderSimulator):
    """
    Add visualization methods to the GerrymanderSimulator class
    
    Returns the enhanced simulator class
    """
    # Add methods to the class
    GerrymanderSimulator.create_district_change_heatmap = create_district_change_heatmap
    GerrymanderSimulator.track_district_stats_over_time = track_district_stats_over_time
    
    return GerrymanderSimulator