"""
Animation capabilities for the GerrymanderSimulator class.
This module adds animation functionalities to capture and save simulation progress.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import shutil
import subprocess
# Add this function to animationCapabilities.py

def capture_animation_frames(self, num_iterations=10000, batch_size=500, 
                           frame_interval=100, output_dir="animation_frames",
                           include_metrics=True, pixels_per_move=20):
    """
    Run the simulation and capture animation frames at regular intervals
    Modified to be compatible with Numba-accelerated code
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add debug prints
    print(f"Capturing frames every {frame_interval} iterations")
    print(f"Output directory: {output_dir}")
    
    frame_filenames = []
    iterations_completed = 0
    
    # Create progress bar
    progress_bar = tqdm(total=num_iterations)
    
    # Set up the figure for the animation frames
    if include_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Capture initial frame
    if include_metrics:
        # Plot initial state with metrics
        plot_animation_metrics_frame(self, fig, axes, 0)
    else:
        # Plot just the district map
        self.plot_districts(ax)
        ax.set_title(f'Iteration 0')
    
    # Save the initial frame
    initial_frame = os.path.join(output_dir, f"frame_000000.png")
    fig.savefig(initial_frame)
    frame_filenames.append(initial_frame)
    
    # Debug check
    print(f"Saved initial frame: {initial_frame}")
    
    # Initialize phase for the simulation
    self.update_phase(0, num_iterations)
    
    # Run the simulation in chunks and capture frames
    while iterations_completed < num_iterations:
        # Determine how many iterations to run before the next frame
        iterations_to_next_frame = min(
            frame_interval,
            num_iterations - iterations_completed
        )
        
        # Create smaller batches to process
        sub_iterations_completed = 0
        while sub_iterations_completed < iterations_to_next_frame:
            current_batch_size = min(batch_size, iterations_to_next_frame - sub_iterations_completed)
            
            # Use existing run_simulation method for a small chunk
            # This avoids Numba compatibility issues
            if self.num_cpus > 1:
                # Use the parallel version
                if self.phase <= 2:
                    # For early phases when we want multi-pixel moves
                    for i in range(0, current_batch_size, pixels_per_move):
                        batch = min(pixels_per_move, current_batch_size - i)
                        self.run_batch_parallel(batch_size=batch)
                        sub_iterations_completed += batch
                else:
                    # For later phases focusing on compactness
                    self.run_batch_parallel(batch_size=current_batch_size)
                    sub_iterations_completed += current_batch_size
            else:
                # Single-threaded processing
                for _ in range(current_batch_size):
                    self.run_iteration()
                    sub_iterations_completed += 1
            
            # Update progress
            progress_bar.update(sub_iterations_completed)
        
        # Update total iterations completed
        iterations_completed += iterations_to_next_frame
        
        # Update phase
        self.update_phase(iterations_completed, num_iterations)
        
        # Capture frame
        self.calculate_all_district_stats()
        
        if include_metrics:
            # Update the metrics plots
            plot_animation_metrics_frame(self, fig, axes, iterations_completed)
        else:
            # Clear the current plot and redraw the map
            ax.clear()
            self.plot_districts(ax)
            ax.set_title(f'Iteration {iterations_completed}')
        
        # Save the frame
        frame_filename = os.path.join(output_dir, f"frame_{iterations_completed:06d}.png")
        fig.savefig(frame_filename)
        frame_filenames.append(frame_filename)
        
        # Force matplotlib to release memory
        plt.close(fig)
        if include_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Print status update
        pop_mean = np.mean(self.district_stats['population'])
        pop_max = np.max(self.district_stats['population'])
        pop_min = np.min(self.district_stats['population'])
        pop_imbalance = (pop_max - pop_min) / pop_mean
        
        red_districts = sum(1 for d in range(self.num_districts) 
                           if self.district_stats['red_votes'][d] > self.district_stats['blue_votes'][d])
        blue_districts = self.num_districts - red_districts
        
        print(f"\nFrame {len(frame_filenames)}, Iteration {iterations_completed}")
        print(f"Population imbalance: {pop_imbalance:.2%}, Min: {pop_min:.0f}, Max: {pop_max:.0f}")
        print(f"Districts: {red_districts} Red, {blue_districts} Blue")
    
    progress_bar.close()
    plt.close()
    
    return frame_filenames
def plot_animation_metrics_frame(simulator, fig, axes, iteration):
    """Plot metrics for an animation frame with enhanced information"""
    # Calculate population statistics
    pop_mean = np.mean(simulator.district_stats['population'])
    pop_max = np.max(simulator.district_stats['population'])
    pop_min = np.min(simulator.district_stats['population'])
    pop_imbalance = (pop_max - pop_min) / pop_mean
    
    # Get district counts
    red_districts = sum(1 for d in range(simulator.num_districts) 
                       if simulator.district_stats['red_votes'][d] > 
                          simulator.district_stats['blue_votes'][d])
    blue_districts = simulator.num_districts - red_districts
    
    # Create title with key metrics
    phase_info = f"Phase: {simulator.phase}" if hasattr(simulator, 'phase') else ""
    fig.suptitle(f'Gerrymandering Simulation - Iteration {iteration}\n' +
                f'{phase_info}, Pop. Imbalance: {pop_imbalance:.2%}, ' +
                f'Districts: {red_districts}R/{blue_districts}B', fontsize=12)
    
    # Plot population by district
    ax = axes[0, 0]
    ax.clear()
    bars = ax.bar(range(1, simulator.num_districts + 1), simulator.district_stats['population'])
    
    # Add horizontal line for mean population
    ax.axhline(y=pop_mean, color='r', linestyle='--', alpha=0.5, 
               label=f'Mean: {pop_mean:.0f}')
    
    # Highlight districts with extreme population
    for i, bar in enumerate(bars):
        if simulator.district_stats['population'][i] > pop_mean * 1.1:
            bar.set_color('red')  # Overpopulated
        elif simulator.district_stats['population'][i] < pop_mean * 0.9:
            bar.set_color('green')  # Underpopulated
    
    ax.set_xlabel('District')
    ax.set_ylabel('Population')
    ax.set_title('Population by District')
    ax.legend()
    
    # Plot compactness (perimeter to area ratios)
    ax = axes[0, 1]
    ax.clear()
    perimeter_to_area = simulator.district_stats['perimeter'] / np.sqrt(simulator.district_stats['area'])
    ax.bar(range(1, simulator.num_districts + 1), perimeter_to_area)
    ax.set_xlabel('District')
    ax.set_ylabel('Perimeter / √Area')
    ax.set_title(f'Compactness by District (Lower is Better)')
    
    # Plot district map
    ax = axes[1, 0]
    ax.clear()
    simulator.plot_districts(ax)
    
    # Plot election results
    ax = axes[1, 1]
    ax.clear()
    simulator.plot_election_results(ax)
    
    fig.tight_layout()
    # Adjust spacing to accommodate the suptitle
    fig.subplots_adjust(top=0.88)

def run_simulation_with_animation(self, num_iterations=10000, batch_size=1000, 
                               frame_interval=100, output_dir="animation_frames", 
                               output_file="simulation_animation.gif", fps=10, 
                               include_metrics=True, delete_frames=True,
                               pixels_per_move=20):
    """
    Run the simulation and create an animation with multi-pixel approach
    Uses existing Numba functions for maximum performance
    """
    print(f"Running simulation with animation for {num_iterations} iterations...")
    print(f"Using {pixels_per_move} pixels per move for faster convergence")
    start_time = time.time()
    
    # Capture frames
    frame_filenames = capture_animation_frames(
        self,
        num_iterations=num_iterations,
        batch_size=batch_size,
        frame_interval=frame_interval,
        output_dir=output_dir,
        include_metrics=include_metrics,
        pixels_per_move=pixels_per_move
    )
    
    # Create animation
    animation_file = create_animation_from_frames(
        frame_filenames,
        output_file=output_file,
        fps=fps
    )
    
    # Clean up frames if requested
    if delete_frames:
        print("Cleaning up individual frames...")
        for filename in frame_filenames:
            try:
                os.remove(filename)
            except:
                pass
        
        # Try to remove the directory
        try:
            os.rmdir(output_dir)
        except:
            pass
    
    end_time = time.time()
    print(f"Simulation and animation completed in {end_time - start_time:.2f} seconds")
    
    return animation_file

def create_animation_from_frames(frame_filenames, output_file="simulation_animation.gif", fps=10):
    """
    Create an animation from saved frames
    
    Parameters:
    - frame_filenames: List of frame filenames
    - output_file: Output video file
    - fps: Frames per second
    
    Returns:
    - The output filename
    """
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        has_ffmpeg = True
    except FileNotFoundError:
        has_ffmpeg = False
        print("Warning: ffmpeg not found, using imageio instead")
    
    if has_ffmpeg:
        # Use ffmpeg to create the animation (much faster)
        print(f"Creating animation with ffmpeg ({len(frame_filenames)} frames)...")
        
        # Ensure output file has .gif extension
        if not output_file.endswith('.gif'):
            output_file = output_file.replace('.mp4', '.gif')
        
        # Use ffmpeg directly to create a gif from the frames
        frame_pattern = os.path.join(os.path.dirname(frame_filenames[0]), 
                                    f"frame_%06d.png")
        
        subprocess.run([
            'ffmpeg', '-y', '-r', str(fps), 
            '-i', frame_pattern,
            '-vf', 'scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
            output_file
        ])
    else:
        # Use imageio as a fallback (more reliable than matplotlib for this)
        try:
            import imageio
            print(f"Creating animation with imageio ({len(frame_filenames)} frames)...")
            
            # Ensure output file has .gif extension
            if not output_file.endswith('.gif'):
                output_file = output_file.replace('.mp4', '.gif')
            
            # Read all images
            images = []
            for filename in frame_filenames:
                images.append(imageio.imread(filename))
            
            # Save as GIF
            imageio.mimsave(output_file, images, fps=fps)
        except Exception as e:
            print(f"Failed to create animation with imageio: {e}")
            # As a last resort, just save the individual frames
            output_dir = "animation_frames_final"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving individual frames to {output_dir}/")
            import shutil
            for i, frame in enumerate(frame_filenames):
                shutil.copy2(frame, f"{output_dir}/frame_{i:04d}.png")
            output_file = output_dir
    
    print(f"Animation saved to {output_file}")
    return output_file

def add_animation_methods(GerrymanderSimulator):
    """
    Add animation methods to the GerrymanderSimulator class
    
    Returns the enhanced simulator class
    """
    # Add methods to the class
    GerrymanderSimulator.capture_animation_frames = capture_animation_frames
    GerrymanderSimulator.run_simulation_with_animation = run_simulation_with_animation
    
    return GerrymanderSimulator