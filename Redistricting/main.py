import numpy as np
from gerrymanderingSimulator import GerrymanderSimulator
from animationCapabilities import add_animation_methods

from visualization import add_visualization_methods  # Use direct import if in same directory
import os

# Load your state map
state_map = np.load('pa_election_data.npy')
# Flip the state map to correct orientation
state_map = np.flip(state_map, axis=0)
print("Flipped state map orientation")

# Create a modified version with better parallelization for many cores
def optimize_for_many_cores(simulator_class):
    original_init = simulator_class.__init__
    
    def enhanced_init(self, state_map, num_districts=13, use_gpu=False):
        # Call original init but override num_cpus calculation
        original_init(self, state_map, num_districts, use_gpu)
        
        # For systems with many cores, limit to a reasonable number
        # Too many processes can create overhead and slow things down
        self.num_cpus = min(32, max(1, os.cpu_count() - 4))
        print(f"Optimized for high-core system: Using {self.num_cpus} CPU cores for parallelization")
        
    simulator_class.__init__ = enhanced_init
    return simulator_class

# Enhance the simulator with animation, visualization, and core optimization
EnhancedSimulator = optimize_for_many_cores(add_animation_methods(add_visualization_methods(GerrymanderSimulator)))

# Initialize with optimized parameters
simulator = EnhancedSimulator(state_map, num_districts=17,use_gpu=True)

# Set up the simulation parameters
simulator.set_target_vote_distribution('fair')

# Run with optimized parameters
simulator.run_simulation_with_animation(
    num_iterations=10000,  # Reduced from 200,000
    batch_size=2000,       # Larger batches for better parallelization
    frame_interval=1000,   # Capture fewer frames to reduce overhead
    include_metrics=False, # Disable metrics during main run for speed
    pixels_per_move=50,    # Increase from default 20 for faster convergence
    fps=5                  # Lower fps for smaller output file
    
)

# After the animation completes, generate a final metrics visualization
simulator.plot_metrics()