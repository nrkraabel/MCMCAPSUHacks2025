import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from gerrymanderingSimulator import GerrymanderSimulator

def fix_orientation(state_map):
    """
    Fix the orientation of the state map if it appears flipped.
    Corrects the common issue where north/south is flipped due to 
    coordinate system differences.
    
    Parameters:
    - state_map: The 3D numpy array with [population, red_votes, blue_votes]
    
    Returns:
    - Corrected state map
    """
    # Flip the map vertically (rows are flipped)
    # This addresses the north/south orientation issue that often occurs
    # because GIS data typically uses a coordinate system where Y increases upward,
    # while image arrays have Y increasing downward
    corrected_map = np.flip(state_map, axis=0)
    return corrected_map

def run_simulation(state_map_path, num_districts=17, scenario='fair', iterations=50000, fix_map_orientation=True):
    """
    Load a processed state map and run the gerrymandering simulation.
    
    Parameters:
    - state_map_path: Path to the numpy state map file
    - num_districts: Number of congressional districts (17 for Pennsylvania)
    - scenario: Gerrymandering scenario ('fair', 'red_gerrymander', 'blue_gerrymander', 'incumbent')
    - iterations: Number of iterations to run the simulation
    - fix_map_orientation: Whether to fix the map orientation (flip north/south)
    
    Returns:
    - The simulator object after running
    """
    # Load the state map
    print(f"Loading state map from {state_map_path}...")
    state_map = np.load(state_map_path)
    
    # Check if we need to fix orientation
    if fix_map_orientation:
        print("Fixing map orientation (flipping north/south)...")
        state_map = fix_orientation(state_map)
    
    # Create simulator
    print(f"Creating simulator with {num_districts} districts...")
    simulator = GerrymanderSimulator(state_map, num_districts=num_districts)
    
    # Set target distribution
    print(f"Setting target vote distribution: {scenario}")
    simulator.set_target_vote_distribution(scenario)
    
    # Plot initial state
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    simulator.plot_districts()
    plt.title('Initial Random Districts')
    
    plt.subplot(2, 2, 2)
    simulator.plot_election_results()
    plt.title('Initial Election Results')
    
    # Run simulation
    print(f"Running simulation for {iterations} iterations...")
    simulator.run_simulation(num_iterations=iterations)
    
    # Plot final state
    plt.subplot(2, 2, 3)
    simulator.plot_districts()
    plt.title('Final Optimized Districts')
    
    plt.subplot(2, 2, 4)
    simulator.plot_election_results()
    plt.title('Final Election Results')
    
    plt.tight_layout()
    plt.savefig("pa_simulation_results.png")
    plt.show()
    
    # Plot detailed metrics
    simulator.plot_metrics()
    
    return simulator

def analyze_fairness(simulator):
    """
    Analyze the fairness of the districting plan.
    Provides statistics on seats vs votes, efficiency gap, etc.
    
    Parameters:
    - simulator: The GerrymanderSimulator instance after running
    """
    # Calculate statewide vote shares
    total_red = np.sum(simulator.state_map[:,:,1])
    total_blue = np.sum(simulator.state_map[:,:,2])
    total_votes = total_red + total_blue
    
    red_vote_pct = total_red / total_votes * 100
    blue_vote_pct = total_blue / total_votes * 100
    
    # Calculate district outcomes
    district_winners = []
    vote_margins = []
    
    for district_id in range(simulator.num_districts):
        red = simulator.district_stats['red_votes'][district_id]
        blue = simulator.district_stats['blue_votes'][district_id]
        
        if red > blue:
            district_winners.append('Red')
        else:
            district_winners.append('Blue')
        
        if red + blue > 0:
            margin = red / (red + blue)
            vote_margins.append(margin)
        else:
            vote_margins.append(0.5)
    
    # Count seats
    red_seats = district_winners.count('Red')
    blue_seats = district_winners.count('Blue')
    
    red_seat_pct = red_seats / simulator.num_districts * 100
    blue_seat_pct = blue_seats / simulator.num_districts * 100
    
    # Calculate the efficiency gap
    # (Measures the partisan advantage through "wasted votes")
    red_wasted = 0
    blue_wasted = 0
    
    for district_id in range(simulator.num_districts):
        red = simulator.district_stats['red_votes'][district_id]
        blue = simulator.district_stats['blue_votes'][district_id]
        
        if red > blue:  # Red wins
            red_wasted += red - ((red + blue) / 2 + 1)  # Excess votes
            blue_wasted += blue  # All losing votes are wasted
        else:  # Blue wins
            blue_wasted += blue - ((red + blue) / 2 + 1)  # Excess votes
            red_wasted += red  # All losing votes are wasted
    
    # Calculate efficiency gap as percentage of total votes
    efficiency_gap = (red_wasted - blue_wasted) / total_votes * 100
    
    # Calculate seats-votes proportionality
    proportional_red_seats = (red_vote_pct / 100) * simulator.num_districts
    seats_votes_asymmetry = red_seats - proportional_red_seats
    
    # Print the analysis
    print("\n" + "="*50)
    print("DISTRICTING FAIRNESS ANALYSIS")
    print("="*50)
    
    print(f"\nStatewide Vote Totals:")
    print(f"  Republican: {red_vote_pct:.1f}% ({total_red:,} votes)")
    print(f"  Democratic: {blue_vote_pct:.1f}% ({total_blue:,} votes)")
    
    print(f"\nDistrict Outcomes:")
    print(f"  Republican: {red_seats} seats ({red_seat_pct:.1f}%)")
    print(f"  Democratic: {blue_seats} seats ({blue_seat_pct:.1f}%)")
    
    print(f"\nSeats-Votes Proportionality:")
    print(f"  Proportional Republican seats: {proportional_red_seats:.2f}")
    print(f"  Actual Republican seats: {red_seats}")
    print(f"  Asymmetry: {seats_votes_asymmetry:.2f} seats")
    
    print(f"\nEfficiency Gap:")
    print(f"  {efficiency_gap:.2f}%")
    if abs(efficiency_gap) > 7:
        advantage = "Republican" if efficiency_gap > 0 else "Democratic"
        print(f"  * High efficiency gap favoring {advantage} party")
    else:
        print(f"  * Efficiency gap within reasonable bounds")
    
    # Count competitive districts (margin within 5%)
    competitive = sum(1 for margin in vote_margins if 0.45 <= margin <= 0.55)
    print(f"\nCompetitive Districts: {competitive} of {simulator.num_districts}")
    
    # Calculate mean-median difference
    mean_margin = np.mean(vote_margins)
    median_margin = np.median(vote_margins)
    mean_median = (mean_margin - median_margin) * 100  # As percentage points
    
    print(f"\nMean-Median Difference: {mean_median:.2f}%")
    if abs(mean_median) > 2:
        advantage = "Republican" if mean_median < 0 else "Democratic"
        print(f"  * Advantage to {advantage} party")
    
    print("="*50)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run gerrymandering simulation on Pennsylvania data")
    parser.add_argument("--input", type=str, default="pa_election_data.npy", 
                      help="Path to the processed state map numpy file")
    parser.add_argument("--districts", type=int, default=17, 
                      help="Number of congressional districts (17 for Pennsylvania)")
    parser.add_argument("--scenario", type=str, default="fair", 
                      choices=["fair", "red_gerrymander", "blue_gerrymander", "incumbent"],
                      help="Gerrymandering scenario to simulate")
    parser.add_argument("--iterations", type=int, default=5000, 
                      help="Number of iterations to run")
    parser.add_argument("--no-flip", action="store_true", 
                      help="Don't flip the map orientation")
    
    args = parser.parse_args()
    
    # Make sure the input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)
    
    # Make sure GerrymanderSimulator is available
    try:
        from gerrymanderingSimulator import GerrymanderSimulator
    except ImportError:
        print("Error: GerrymanderSimulator class not found.")
        print("Make sure GerrymanderSimulator.py is in the current directory or PYTHONPATH.")
        sys.exit(1)
    
    # Run the simulation
    simulator = run_simulation(
        args.input, 
        num_districts=args.districts,
        scenario=args.scenario,
        iterations=args.iterations,
        fix_map_orientation=not args.no_flip
    )
    
    # Analyze the fairness of the districting plan
    analyze_fairness(simulator)
    
    print("\nSimulation complete!")
    print("Results saved to pa_simulation_results.png")

if __name__ == "__main__":
    main()