import numpy as np
import os
from gerrymanderingSimulator import GerrymanderSimulator
from animationCapabilities import add_animation_methods
from visualization import add_visualization_methods

# Load your state map
state_map = np.load('/storage/home/nrk5343/work/MCMCAPSUHacks2025/pa_election_data.npy')
# Flip the state map to correct orientation
state_map = np.flip(state_map, axis=0)
print("Flipped state map orientation")

# Create a modified version with better parallelization for many cores
def optimize_for_slurm(simulator_class):
    original_init = simulator_class.__init__
    
    def enhanced_init(self, state_map, num_districts=13, use_gpu=False):
        # Call original init
        original_init(self, state_map, num_districts, use_gpu)
        
     
        # For systems with many cores, limit to a reasonable number
        self.num_cpus = min(64, max(1, os.cpu_count() - 1))
        print(f"SLURM CPU count not found, using {self.num_cpus} CPU cores")
        
        # Disable GPU as it's often not available or configured correctly on clusters
        self.use_gpu = False
        
    simulator_class.__init__ = enhanced_init
    return simulator_class

# Enhance the simulator with Slurm-aware optimization
EnhancedSimulator = optimize_for_slurm(add_visualization_methods(GerrymanderSimulator))

# Initialize with optimized parameters
simulator = EnhancedSimulator(state_map, num_districts=17)

# Set up the simulation parameters
simulator.set_target_vote_distribution('fair')

# Run without animation for better performance on cluster
print("Starting simulation run...")
simulator.run_simulation(
    num_iterations=20000,  # Can use more iterations on a cluster
    batch_size=5000,       # Larger batches for better parallelization
    use_parallel=True,     # Explicitly enable parallel processing
    pixels_per_move=40     # Process more pixels per move for efficiency
)

# Save results
simulator.save_district_map("final_district_map.npy")
simulator.export_district_stats("final_district_stats.csv")

# Generate plots (optional - may not work on headless cluster)
try:
    fig = simulator.plot_metrics()
    fig.savefig("final_metrics.png", dpi=300)
    print("Saved metrics visualization to final_metrics.png")
except Exception as e:
    print(f"Could not generate visualization: {e}")
    print("Continuing with results export")

print("Simulation complete. Results saved.")