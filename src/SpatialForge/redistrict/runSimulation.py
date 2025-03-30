"""
Command-line interface for running gerrymandering simulations with animations.

This module provides a command-line interface and high-level functions for running
gerrymandering simulations with various visualization options.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from gerrymanderingSimulator import GerrymanderSimulator
from animationCapabilities import add_animation_methods
from MCMCAPSUHacks2025.Redistricting.visualization import add_visualization_methods, plot_metrics_evolution, compare_scenarios_metrics, create_comparison_visualization
import subprocess
from tqdm import tqdm

def run_gerrymandering_animation(state_map_path, num_districts=13, scenario='fair', 
                               iterations=10000, frame_interval=100, fps=10,
                               fix_map_orientation=True, use_parallel=True, use_gpu=False):
    """
    Run a gerrymandering simulation with animation
    
    Parameters:
    - state_map_path: Path to the numpy state map file
    - num_districts: Number of congressional districts
    - scenario: Gerrymandering scenario ('fair', 'red_gerrymander', 'blue_gerrymander', 'incumbent')
    - iterations: Number of iterations
    - frame_interval: Save a frame every N iterations
    - fps: Frames per second in the output video
    - fix_map_orientation: Whether to fix the map orientation (flip north/south)
    - use_parallel: Whether to use parallel processing with multiple CPUs
    - use_gpu: Whether to use GPU acceleration if available
    
    Returns:
    - Paths to created files (animation, heatmap, stats CSV)
    """
    # Load the state map
    print(f"Loading state map from {state_map_path}...")
    state_map = np.load(state_map_path)
    
    # Check if we need to fix orientation
    if fix_map_orientation:
        print("Fixing map orientation (flipping north/south)...")
        state_map = np.flip(state_map, axis=0)
    
    # Enhance the GerrymanderSimulator class
    EnhancedGerrymanderSimulator = add_visualization_methods(add_animation_methods(GerrymanderSimulator))
    
    # Create enhanced simulator
    print(f"Creating enhanced simulator with {num_districts} districts...")
    simulator = EnhancedGerrymanderSimulator(state_map, num_districts=num_districts, use_gpu=use_gpu)
    
    # Set target distribution
    print(f"Setting target vote distribution: {scenario}")
    simulator.set_target_vote_distribution(scenario)
    
    # Plot initial state
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    simulator.plot_districts()
    plt.title('Initial Random Districts')
    
    plt.subplot(1, 2, 2)
    simulator.plot_election_results()
    plt.title('Initial Election Results')
    
    plt.tight_layout()
    plt.savefig("initial_districts.png")
    plt.close()
    
    # Run simulation with animation
    start_time = time.time()
    animation_file = simulator.run_simulation_with_animation(
        num_iterations=iterations,
        frame_interval=frame_interval,
        fps=fps,
        output_dir=f"animation_frames_{scenario}",
        output_file=f"simulation_animation_{scenario}.mp4"
    )
    
    # Create a heatmap of district changes
    heatmap_file = simulator.create_district_change_heatmap(
        num_iterations=iterations // 10,  # Run a shorter simulation for the heatmap
        sample_interval=frame_interval,
        output_file=f"district_changes_{scenario}.png"
    )
    
    # Track district stats over time
    stats_file, stats_df = simulator.track_district_stats_over_time(
        num_iterations=iterations,
        sample_interval=frame_interval * 5,  # Sample less frequently for stats
        output_file=f"district_stats_{scenario}.csv"
    )
    
    # Plot the final state
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    simulator.plot_districts()
    plt.title('Final Optimized Districts')
    
    plt.subplot(1, 2, 2)
    simulator.plot_election_results()
    plt.title('Final Election Results')
    
    plt.tight_layout()
    final_districts_file = f"final_districts_{scenario}.png"
    plt.savefig(final_districts_file)
    plt.close()
    
    # Plot detailed metrics
    metrics_fig = simulator.plot_metrics()
    metrics_file = f"metrics_{scenario}.png"
    plt.savefig(metrics_file)
    plt.close()
    
    # Plot evolution of metrics
    metrics_evolution_file = plot_metrics_evolution(stats_df, simulator, f"metrics_evolution_{scenario}.png")
    
    # Report on the time taken
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Return paths to created files
    return {
        'animation': animation_file,
        'heatmap': heatmap_file,
        'stats': stats_file,
        'initial_districts': "initial_districts.png",
        'final_districts': final_districts_file,
        'metrics': metrics_file,
        'metrics_evolution': metrics_evolution_file
    }

def compare_gerrymandering_scenarios(state_map_path, num_districts=13, iterations=10000, 
                                    frame_interval=100, fps=10,
                                    fix_map_orientation=True, use_parallel=True, use_gpu=False):
    """
    Run simulations for different gerrymandering scenarios and create a comparison
    
    Parameters:
    - state_map_path: Path to the numpy state map file
    - num_districts: Number of congressional districts
    - iterations: Number of iterations
    - frame_interval: Save a frame every N iterations
    - fps: Frames per second in the output video
    - fix_map_orientation: Whether to fix the map orientation (flip north/south)
    - use_parallel: Whether to use parallel processing with multiple CPUs
    - use_gpu: Whether to use GPU acceleration if available
    
    Returns:
    - Dictionary with paths to all created files
    """
    scenarios = ['fair', 'red_gerrymander', 'blue_gerrymander', 'incumbent']
    results = {}
    
    # Run simulations for each scenario
    for scenario in scenarios:
        print(f"\n\n===== Running {scenario} scenario =====\n")
        results[scenario] = run_gerrymandering_animation(
            state_map_path=state_map_path,
            num_districts=num_districts,
            scenario=scenario,
            iterations=iterations,
            frame_interval=frame_interval,
            fps=fps,
            fix_map_orientation=fix_map_orientation,
            use_parallel=use_parallel,
            use_gpu=use_gpu
        )
    
    # Create a comparison visualization of the final districts and heatmaps
    comparison_file = create_comparison_visualization(results)
    
    # Load stats data for all scenarios
    all_stats = {}
    for scenario in scenarios:
        all_stats[scenario] = pd.read_csv(results[scenario]['stats'])
    
    # Create a comparison of key metrics
    metrics_comparison_file = compare_scenarios_metrics(all_stats, num_districts)
    
    # Try to create a combined animation if ffmpeg is available
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        has_ffmpeg = True
        
        # Create a grid of animations
        print("Creating combined animations grid...")
        
        # Create a command to combine all animations side by side
        cmd = [
            'ffmpeg', '-y',
            '-i', results['fair']['animation'],
            '-i', results['red_gerrymander']['animation'],
            '-i', results['blue_gerrymander']['animation'],
            '-i', results['incumbent']['animation'],
            '-filter_complex', 
            '[0:v]setpts=PTS-STARTPTS, pad=iw*2:ih*2:0:0[a]; ' +
            '[1:v]setpts=PTS-STARTPTS, pad=iw*2:ih*2:iw:0[b]; ' +
            '[2:v]setpts=PTS-STARTPTS, pad=iw*2:ih*2:0:ih[c]; ' +
            '[3:v]setpts=PTS-STARTPTS, pad=iw*2:ih*2:iw:ih[d]; ' +
            '[a][b]overlay=w[top]; ' +
            '[c][d]overlay=w[bottom]; ' +
            '[top][bottom]overlay=0:H/2[out]',
            '-map', '[out]',
            'comparison_grid.mp4'
        ]
        
        subprocess.run(cmd)
        print("Combined animation grid saved to comparison_grid.mp4")
        
        results['combined_animation'] = 'comparison_grid.mp4'
    except:
        print("ffmpeg not available, skipping combined animation creation")
    
    results['comparison_image'] = comparison_file
    results['metrics_comparison'] = metrics_comparison_file
    
    return results

def main():
    """
    Main entry point for running gerrymandering simulations with animations
    """
    parser = argparse.ArgumentParser(description='Gerrymandering Simulation with Animation')
    
    parser.add_argument('--state_map', type=str, required=True,
                        help='Path to the numpy state map file (.npy)')
    
    parser.add_argument('--num_districts', type=int, default=13,
                        help='Number of congressional districts (default: 13)')
    
    parser.add_argument('--scenario', type=str, default='fair',
                        choices=['fair', 'red_gerrymander', 'blue_gerrymander', 'incumbent', 'all'],
                        help='Gerrymandering scenario (default: fair)')
    
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of iterations to run (default: 10000)')
    
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of animation frames to capture (default: 100)')
    
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second in the output video (default: 10)')
    
    parser.add_argument('--no_parallel', action='store_true',
                        help='Disable parallel processing')
    
    parser.add_argument('--use_gpu', action='store_true',
                        help='Enable GPU acceleration if available')
    
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output files (default: output)')
    
    parser.add_argument('--no_animation', action='store_true',
                        help='Disable animation generation (faster)')
    
    parser.add_argument('--interactive', action='store_true',
                        help='Enable interactive mode to pause and resume the simulation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save original working directory and change to output directory
    original_dir = os.getcwd()
    os.chdir(args.output_dir)
    
    try:
        # Calculate frame interval based on number of frames
        frame_interval = args.iterations // args.frames
        
        if args.scenario == 'all':
            # Run comparison of all scenarios
            results = compare_gerrymandering_scenarios(
                state_map_path=args.state_map,
                num_districts=args.num_districts,
                iterations=args.iterations,
                frame_interval=frame_interval,
                fps=args.fps,
                use_parallel=not args.no_parallel,
                use_gpu=args.use_gpu
            )
            
            print("\nGenerated files:")
            for scenario, files in results.items():
                if isinstance(files, dict):
                    print(f"\n{scenario.upper()} scenario outputs:")
                    for file_type, file_path in files.items():
                        print(f"  - {file_type}: {file_path}")
                else:
                    print(f"  - {scenario}: {files}")
        else:
            # Run a single scenario
            if args.no_animation:
                # Just run the simulation without animation
                # Load the state map
                print(f"Loading state map from {args.state_map}...")
                state_map = np.load(args.state_map)
                
                # Fix orientation if needed
                state_map = np.flip(state_map, axis=0)
                
                # Enhance the GerrymanderSimulator class
                EnhancedGerrymanderSimulator = add_visualization_methods(add_animation_methods(GerrymanderSimulator))
                
                # Create simulator
                print(f"Creating simulator with {args.num_districts} districts...")
                simulator = EnhancedGerrymanderSimulator(
                    state_map, 
                    num_districts=args.num_districts, 
                    use_gpu=args.use_gpu
                )
                
                # Set target distribution
                print(f"Setting target vote distribution: {args.scenario}")
                simulator.set_target_vote_distribution(args.scenario)
                
                # Plot initial state
                plt.figure(figsize=(15, 10))
                plt.subplot(1, 2, 1)
                simulator.plot_districts()
                plt.title('Initial Random Districts')
                
                plt.subplot(1, 2, 2)
                simulator.plot_election_results()
                plt.title('Initial Election Results')
                
                plt.tight_layout()
                initial_file = "initial_districts.png"
                plt.savefig(initial_file)
                plt.close()
                
                # Run simulation
                print(f"Running simulation for {args.iterations} iterations...")
                if args.interactive:
                    # Run in smaller batches with pauses
                    batch_size = min(1000, args.iterations // 10)
                    remaining = args.iterations
                    
                    while remaining > 0:
                        current_batch = min(batch_size, remaining)
                        
                        # Run a batch
                        if not args.no_parallel:
                            simulator.run_batch_parallel(batch_size=current_batch)
                        else:
                            for _ in tqdm(range(current_batch)):
                                simulator.run_iteration()
                                simulator.temperature *= simulator.cooling_rate
                        
                        # Update remaining
                        remaining -= current_batch
                        
                        # Calculate and print current stats
                        simulator.calculate_all_district_stats()
                        score = simulator.score_map()
                        
                        # Calculate election results
                        red_districts = sum(1 for d in range(simulator.num_districts) 
                                         if simulator.district_stats['red_votes'][d] > 
                                            simulator.district_stats['blue_votes'][d])
                        blue_districts = simulator.num_districts - red_districts
                        
                        print(f"\nCompleted {args.iterations - remaining}/{args.iterations} iterations")
                        print(f"Current score: {score:.2f}")
                        print(f"Districts: {red_districts} Red, {blue_districts} Blue")
                        
                        # Plot current state
                        plt.figure(figsize=(15, 10))
                        plt.subplot(1, 2, 1)
                        simulator.plot_districts()
                        plt.title(f'Districts after {args.iterations - remaining} iterations')
                        
                        plt.subplot(1, 2, 2)
                        simulator.plot_election_results()
                        plt.title('Current Election Results')
                        
                        plt.tight_layout()
                        plt.savefig(f"districts_iter_{args.iterations - remaining}.png")
                        plt.show()
                        
                        # Ask to continue
                        if remaining > 0:
                            input("\nPress Enter to continue...")
                else:
                    # Run the simulation in one go
                    simulator.run_simulation(
                        num_iterations=args.iterations,
                        use_parallel=not args.no_parallel
                    )
                
                # Plot final results
                plt.figure(figsize=(15, 10))
                plt.subplot(1, 2, 1)
                simulator.plot_districts()
                plt.title('Final Optimized Districts')
                
                plt.subplot(1, 2, 2)
                simulator.plot_election_results()
                plt.title('Final Election Results')
                
                plt.tight_layout()
                final_file = "final_districts.png"
                plt.savefig(final_file)
                plt.close()
                
                # Plot metrics
                metrics_fig = simulator.plot_metrics()
                metrics_file = "metrics.png"
                plt.savefig(metrics_file)
                plt.close()
                
                # Save district map and stats
                simulator.save_district_map("district_map.npy")
                simulator.export_district_stats("district_stats.csv")
                
                print("\nGenerated files:")
                print(f"  - Initial districts: {initial_file}")
                print(f"  - Final districts: {final_file}")
                print(f"  - Metrics: {metrics_file}")
                print(f"  - District map: district_map.npy")
                print(f"  - District stats: district_stats.csv")
            else:
                # Run with animation
                results = run_gerrymandering_animation(
                    state_map_path=args.state_map,
                    num_districts=args.num_districts,
                    scenario=args.scenario,
                    iterations=args.iterations,
                    frame_interval=frame_interval,
                    fps=args.fps,
                    use_parallel=not args.no_parallel,
                    use_gpu=args.use_gpu
                )
                
                print("\nGenerated files:")
                for file_type, file_path in results.items():
                    print(f"  - {file_type}: {file_path}")
    
    finally:
        # Change back to original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main()