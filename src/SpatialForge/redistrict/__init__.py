"""
Gerrymandering Simulator Package

This package provides tools for simulating and visualizing the gerrymandering
process using a Markov Chain Monte Carlo approach with simulated annealing.

Modules:
    - gerrymandering_simulator: Core simulation functionality
    - animation_capabilities: Tools for creating animations of the simulation
    - visualization: Advanced visualization utilities
    - run_simulation: Command-line interface and high-level functions

Example usage:
    from gerrymandering_simulator import GerrymanderSimulator
    from animation_capabilities import add_animation_methods
    from visualization import add_visualization_methods
    
    # Load a state map
    state_map = np.load('state_map.npy')
    
    # Create an enhanced simulator
    EnhancedSimulator = add_visualization_methods(add_animation_methods(GerrymanderSimulator))
    simulator = EnhancedSimulator(state_map, num_districts=13)
    
    # Set target distribution and run simulation
    simulator.set_target_vote_distribution('fair')
    simulator.run_simulation_with_animation(num_iterations=10000)
"""

from .gerrymandering_simulator import GerrymanderSimulator
from .animation_capabilities import add_animation_methods
from ..core.post.visualization import add_visualization_methods, plot_metrics_evolution, compare_scenarios_metrics
from .run_imulation import run_gerrymandering_animation, compare_gerrymandering_scenarios, main

__all__ = [
    'GerrymanderSimulator',
    'add_animation_methods',
    'add_visualization_methods',
    'plot_metrics_evolution',
    'compare_scenarios_metrics',
    'run_gerrymandering_animation',
    'compare_gerrymandering_scenarios',
    'main'
]