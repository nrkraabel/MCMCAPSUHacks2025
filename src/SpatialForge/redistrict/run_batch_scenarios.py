import numpy as np
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

# Import your simulator classes - adjust paths if needed
from gerrymanderingSimulator import GerrymanderSimulator
from animationCapabilities import add_animation_methods
from visualization import add_visualization_methods

def run_single_scenario(scenario_config, base_output_dir, enable_animation=True):
    """
    Run a single gerrymandering scenario with the given configuration and save animations
    
    Parameters:
    - scenario_config: Dict containing scenario parameters
    - base_output_dir: Base directory to save outputs
    - enable_animation: Whether to generate animations
    
    Returns:
    - Path to the output directory with results
    """
    scenario_name = scenario_config['name']
    print(f"\n{'='*80}\nRunning scenario: {scenario_name}\n{'='*80}")
    
    # Create output directory for this scenario
    scenario_dir = os.path.join(base_output_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Load the state map
    state_map = np.load(scenario_config['state_map_path'])
    if scenario_config.get('flip_map', False):
        state_map = np.flip(state_map, axis=0)
        print("Flipped state map orientation")
    
    # Get number of districts
    num_districts = scenario_config.get('num_districts', 13)
    EnhancedSimulator = add_visualization_methods(GerrymanderSimulator)
    # Create an enhanced simulator with animation capabilities
    EnhancedSimulator = add_animation_methods(EnhancedSimulator)
    simulator = EnhancedSimulator(
        state_map, 
        num_districts=num_districts,
        use_gpu=scenario_config.get('use_gpu', False)
    )
    
    # Set target vote distribution
    distribution_type = scenario_config.get('distribution_type', 'fair')
    red_proportion = scenario_config.get('red_proportion', None)
    simulator.set_target_vote_distribution(distribution_type, red_proportion)
    
    # Save initial state
    simulator.save_district_map(os.path.join(scenario_dir, "initial_district_map.npy"))
    simulator.export_district_stats(os.path.join(scenario_dir, "initial_district_stats.csv"))
    
    # Create initial visualization
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = fig.add_subplot(121)
    simulator.plot_districts(ax=ax1, show_stats=True)
    
    ax2 = fig.add_subplot(122)
    simulator.plot_election_results(ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(scenario_dir, "initial_districts.png"), dpi=300)
    plt.close(fig)
    
    # Get simulation parameters
    num_iterations = scenario_config.get('num_iterations', 10000)
    batch_size = scenario_config.get('batch_size', 1000)
    use_parallel = scenario_config.get('use_parallel', True)
    pixels_per_move = scenario_config.get('pixels_per_move', 40)
    
    # Set up animation if enabled
    if enable_animation:
        animation_frames = scenario_config.get('animation_frames', 150)
        animation_interval = scenario_config.get('animation_interval', 200)
        animation_file = os.path.join(scenario_dir, f"{scenario_name}_animation.mp4")
        
        # Set up the animation - no need to pass num_iterations here
        simulator.setup_animation(save_path=animation_file, frames=animation_frames, 
                                 interval=animation_interval, dpi=300)
    
    # Run the simulation
    print(f"Starting simulation for scenario '{scenario_name}'...")
    
    # Run simulation with animation capture
    if enable_animation:
        simulator.run_simulation(
            num_iterations=num_iterations,
            batch_size=batch_size,
            use_parallel=use_parallel,
            pixels_per_move=pixels_per_move,
            capture_animation=True,
            animation_frames=animation_frames
        )
    else:
        simulator.run_simulation(
            num_iterations=num_iterations,
            batch_size=batch_size,
            use_parallel=use_parallel,
            pixels_per_move=pixels_per_move
        )
    
    # Save final results
    district_map_path = os.path.join(scenario_dir, "final_district_map.npy")
    stats_path = os.path.join(scenario_dir, "final_district_stats.csv")
    
    simulator.save_district_map(district_map_path)
    simulator.export_district_stats(stats_path)
    
    # Generate district change heatmap if enabled in config
    if scenario_config.get('generate_heatmap', False):
        heatmap_path = os.path.join(scenario_dir, f"{scenario_name}_changes_heatmap.png")
        simulator.create_district_change_heatmap(
            num_iterations=min(10000, num_iterations // 2),  # Use shorter run for heatmap
            batch_size=batch_size,
            sample_interval=100,
            output_file=heatmap_path
        )
    
    # Track district stats evolution if enabled in config
    if scenario_config.get('track_stats_evolution', False):
        stats_csv_path = os.path.join(scenario_dir, f"{scenario_name}_stats_evolution.csv")
        metrics_plot_path = os.path.join(scenario_dir, f"{scenario_name}_metrics_evolution.png")
        
        output_file, stats_df = simulator.track_district_stats_over_time(
            num_iterations=min(10000, num_iterations // 2),  # Use shorter run for tracking
            batch_size=batch_size,
            sample_interval=200,
            output_file=stats_csv_path
        )
        
        # Plot the metrics evolution
        if hasattr(simulator, 'plot_metrics_evolution'):
            simulator.plot_metrics_evolution(stats_df, simulator, output_file=metrics_plot_path)
    
    # Generate final visualization
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = fig.add_subplot(121)
    simulator.plot_districts(ax=ax1, show_stats=True)
    
    ax2 = fig.add_subplot(122)
    simulator.plot_election_results(ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(scenario_dir, "final_districts.png"), dpi=300)
    plt.close(fig)
    
    print(f"Scenario '{scenario_name}' completed. Results saved to {scenario_dir}")
    return scenario_dir

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run gerrymandering simulation scenarios with animations')
    parser.add_argument('--config', type=str, required=True, help='Path to scenario configuration JSON file')
    parser.add_argument('--output-dir', type=str, default='./results', help='Base directory for output files')
    parser.add_argument('--scenario', type=str, help='Specific scenario to run (by name)')
    parser.add_argument('--no-animation', action='store_true', help='Disable animation generation')
    parser.add_argument('--parallel', action='store_true', help='Run scenarios in parallel (uses all available cores)')
    args = parser.parse_args()
    
    # Read the configuration file
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create base output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a copy of the configuration file
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare scenarios to run
    scenarios_to_run = []
    if args.scenario:
        # Run only the specified scenario
        for scenario in config['scenarios']:
            if scenario['name'] == args.scenario:
                scenarios_to_run.append(scenario)
                break
        if not scenarios_to_run:
            print(f"Error: Scenario '{args.scenario}' not found in configuration file.")
            sys.exit(1)
    else:
        # Run all scenarios
        scenarios_to_run = config['scenarios']
    
    # Run scenarios
    results = {}
    
    if args.parallel and len(scenarios_to_run) > 1:
        # Run scenarios in parallel
        print(f"Running {len(scenarios_to_run)} scenarios in parallel...")
        
        # Create process pool using all available cores
        num_processes = min(len(scenarios_to_run), mp.cpu_count())
        pool = mp.Pool(processes=num_processes)
        
        # Create a partial function with fixed arguments
        run_func = partial(run_single_scenario, 
                          base_output_dir=output_dir, 
                          enable_animation=not args.no_animation)
        
        # Run scenarios in parallel
        scenario_dirs = pool.map(run_func, scenarios_to_run)
        
        # Close the pool
        pool.close()
        pool.join()
        
        # Store results
        for i, scenario_dir in enumerate(scenario_dirs):
            results[scenarios_to_run[i]['name']] = scenario_dir
    else:
        # Run scenarios sequentially
        for scenario in scenarios_to_run:
            scenario_dir = run_single_scenario(
                scenario, 
                output_dir, 
                enable_animation=not args.no_animation
            )
            results[scenario['name']] = scenario_dir
    
    # Print summary of results
    print("\n\n" + "="*40)
    print("SIMULATION BATCH COMPLETE")
    print("="*40)
    print(f"Output directory: {output_dir}")
    print(f"Scenarios completed: {len(results)}")
    for name, path in results.items():
        print(f"  - {name}: {path}")
    print("="*40)

if __name__ == "__main__":
    main()