import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os
from tqdm import tqdm
import time
import subprocess
import glob

def add_animation_methods(simulator_class):
    """
    Extend the GerrymanderSimulator class with animation capabilities optimized for HPC environments
    
    Parameters:
    - simulator_class: The original GerrymanderSimulator class
    
    Returns:
    - Enhanced simulator class with animation methods
    """
    # Create animation method
    def setup_animation(self, save_path=None, frames=100, interval=200, dpi=100):
        """
        Set up an animation of the redistricting process
        
        Parameters:
        - save_path: Path to save the animation (if None, animation will be shown instead)
        - frames: Number of frames to capture
        - interval: Milliseconds between frames
        - dpi: Resolution of the output animation
        
        Returns:
        - fig, anim: Figure and animation objects
        """
        # Store frames as a parameter - we'll calculate the frame interval later
        # when we know the actual number of iterations
        self.animation_frames = []
        self.animation_stats = []
        self.animation_target_frames = frames
        
        # Set up figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Reduced figure size
        
        # Create initial district plot
        self.plot_districts(ax=ax1, show_stats=False)
        ax1.set_title('District Map')
        
        # Create initial election results plot
        self.plot_election_results(ax=ax2)
        ax2.set_title('Election Results by District')
        
        # Add a status text on the figure
        self.animation_status_text = fig.text(0.5, 0.01, 'Initializing...', 
                                              ha='center', va='bottom')
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for status text
        
        self.animation_fig = fig
        self.animation_axes = (ax1, ax2)
        
        # Process the save path
        if save_path:
            # Get directory and base name
            self.animation_dir = os.path.dirname(save_path)
            base_name = os.path.basename(save_path).split('.')[0]
            
            # Create directories for frame-by-frame images
            self.animation_frames_dir = os.path.join(self.animation_dir, f"{base_name}_frames")
            os.makedirs(self.animation_frames_dir, exist_ok=True)
            
            # Set the video path
            self.animation_save_path = save_path
        else:
            self.animation_save_path = None
            self.animation_frames_dir = None
            
        self.animation_interval = interval
        self.animation_dpi = dpi
        
        return fig, (ax1, ax2)
    
    def capture_animation_frame(self):
        """Capture the current state as a frame for the animation"""
        if not hasattr(self, 'animation_fig'):
            return
        
        # Make a deep copy of the current district map and stats
        district_map_copy = self.district_map.copy()
        stats_copy = {k: v.copy() for k, v in self.district_stats.items()}
        
        self.animation_frames.append(district_map_copy)
        self.animation_stats.append(stats_copy)
        
        # If we're saving individual frames, do it now
        if hasattr(self, 'animation_frames_dir') and self.animation_frames_dir:
            frame_num = len(self.animation_frames) - 1
            
            # Update the figure with this frame
            self.update_animation_frame(frame_num)
            
            # Save the frame as PNG
            frame_path = os.path.join(self.animation_frames_dir, f"frame_{frame_num:04d}.png")
            try:
                self.animation_fig.savefig(frame_path, dpi=self.animation_dpi)
                print(f"Saved animation frame {frame_num} to {frame_path}")
            except Exception as e:
                print(f"Error saving frame {frame_num}: {e}")
    
    def update_animation_frame(self, frame_num, *fargs):
        """Update function for animation"""
        if frame_num >= len(self.animation_frames):
            return
        
        # Restore the district map and stats for this frame
        district_map = self.animation_frames[frame_num]
        stats = self.animation_stats[frame_num]
        
        # Create a masked array where invalid areas are masked
        masked_district_map = np.ma.masked_array(
            district_map, 
            mask=~self.valid_mask
        )
        
        # Update district map plot
        ax1 = self.animation_axes[0]
        ax1.clear()
        
        # Plot the district map
        cmap = plt.cm.get_cmap('tab20', self.num_districts)
        cmap.set_bad('white')
        ax1.imshow(masked_district_map, cmap=cmap)
        ax1.axis('off')
        
        # Update election results plot
        ax2 = self.animation_axes[1]
        ax2.clear()
        
        # Calculate vote margins for this frame
        vote_margins = []
        for district_id in range(self.num_districts):
            red = stats['red_votes'][district_id]
            blue = stats['blue_votes'][district_id]
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
        ax2.scatter(district_indices, vote_margins, color='red', label='Actual')
        
        # If we have target margins, plot those too
        if hasattr(self, 'target_vote_margins') and self.target_vote_margins is not None:
            ax2.plot(district_indices, self.target_vote_margins, 'b--', label='Target')
        
        # Add a line at 50%
        ax2.axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_xlabel('District (Ranked by Republican vote share)')
        ax2.set_ylabel('Republican vote share')
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0.5, self.num_districts + 0.5)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Calculate population stats
        pop_mean = np.mean(stats['population'])
        pop_max = np.max(stats['population'])
        pop_min = np.min(stats['population'])
        pop_imbalance = (pop_max - pop_min) / pop_mean
        
        # Get district counts
        red_districts = sum(1 for d in range(self.num_districts) 
                      if stats['red_votes'][d] > stats['blue_votes'][d])
        blue_districts = self.num_districts - red_districts
        
        # Update status text
        # Calculate the approximate iteration number based on frame number
        if hasattr(self, 'animation_frame_interval') and self.animation_frame_interval > 0:
            iteration = frame_num * self.animation_frame_interval
        else:
            iteration = frame_num  # Fallback if interval not known
            
        self.animation_status_text.set_text(
            f'Iteration: {iteration} | Population imbalance: {pop_imbalance:.2%} | ' +
            f'Districts: {red_districts} Red, {blue_districts} Blue | ' +
            f'Phase: {self.phase if hasattr(self, "phase") else "N/A"}'
        )
        
        return (ax1, ax2, self.animation_status_text)
    
    def try_compile_video_from_frames(self):
        """
        Try to compile PNG frames into a video using ffmpeg subprocess with various methods
        Returns True if successful, False otherwise
        """
        if not self.animation_frames_dir or not self.animation_save_path:
            return False
            
        print(f"Attempting to compile video from frames in {self.animation_frames_dir}...")
        
        # Check if ffmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("FFmpeg not found. Cannot compile video.")
            return False
            
        # Get the frames
        frame_pattern = os.path.join(self.animation_frames_dir, "frame_*.png")
        frames = sorted(glob.glob(frame_pattern))
        
        if not frames:
            print("No frames found to compile video.")
            return False
            
        print(f"Found {len(frames)} frames to compile.")
        
        # Try different encoding options for compatibility
        methods = [
            # Method 1: libx264 with yuv420p pixel format (most compatible)
            ['ffmpeg', '-framerate', '10', '-pattern_type', 'glob', '-i', 
             frame_pattern, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 
             '-preset', 'slow', '-crf', '22', '-y', self.animation_save_path],
            
            # Method 2: mpeg4 encoder (older but widely compatible)
            ['ffmpeg', '-framerate', '10', '-pattern_type', 'glob', '-i', 
             frame_pattern, '-c:v', 'mpeg4', '-q:v', '5', '-y', 
             self.animation_save_path],
             
            # Method 3: GIF output if video fails
            ['ffmpeg', '-framerate', '10', '-pattern_type', 'glob', '-i', 
             frame_pattern, '-vf', 'scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse', 
             '-y', self.animation_save_path.replace('.mp4', '.gif')]
        ]
        
        for i, method in enumerate(methods):
            try:
                print(f"Trying video compilation method {i+1}...")
                result = subprocess.run(method, capture_output=True, check=False)
                
                if result.returncode == 0:
                    print(f"Successfully compiled video using method {i+1}.")
                    print(f"Video saved to: {self.animation_save_path}")
                    return True
                else:
                    print(f"Method {i+1} failed with error: {result.stderr.decode()}")
                    
                    # If this was the last method and it was for GIF, check if the GIF was created
                    if i == len(methods) - 1 and method[-1].endswith('.gif') and os.path.exists(method[-1]):
                        print(f"Created GIF animation instead at: {method[-1]}")
                        return True
                        
            except Exception as e:
                print(f"Error with method {i+1}: {e}")
        
        print("All video compilation methods failed. Individual frames are still available.")
        return False
    
    def create_animation(self):
        """Create the final animation from captured frames"""
        if not hasattr(self, 'animation_fig') or len(self.animation_frames) == 0:
            print("No animation data available. Run setup_animation before simulation.")
            return None
        
        print(f"Creating animation with {len(self.animation_frames)} frames...")
        
        # If we've been saving frames all along, try to compile them
        if self.animation_frames_dir and os.path.exists(self.animation_frames_dir):
            success = self.try_compile_video_from_frames()
            if success:
                return True
                
        # If we didn't succeed with pre-saved frames or weren't saving them, 
        # try the matplotlib animation approach
        try:
            # Create the animation
            anim = animation.FuncAnimation(
                self.animation_fig, 
                self.update_animation_frame, 
                frames=len(self.animation_frames),
                interval=self.animation_interval,
                blit=True
            )
            
            # Save or show the animation
            if self.animation_save_path:
                print(f"Attempting to save animation to {self.animation_save_path}...")
                
                # Try to use a more compatible writer setup
                try:
                    # First try with ffmpeg writer with specific args for compatibility
                    writer = animation.FFMpegWriter(
                        fps=10, 
                        metadata=dict(artist='GerrymanderSimulator'),
                        codec='libx264',  # This is usually more compatible
                        bitrate=1000,     # Lower bitrate
                        extra_args=[      # Additional parameters for compatibility
                            '-pix_fmt', 'yuv420p',
                            '-preset', 'slow',
                            '-crf', '22'
                        ]
                    )
                    anim.save(self.animation_save_path, writer=writer, dpi=self.animation_dpi)
                    print(f"Animation saved to {self.animation_save_path}")
                    return anim
                except Exception as e1:
                    print(f"FFMpeg writer error: {e1}")
                    
                    # If that fails, try saving individual frames
                    try:
                        print("Saving individual frames instead...")
                        frames_dir = self.animation_frames_dir or f"{os.path.splitext(self.animation_save_path)[0]}_frames"
                        os.makedirs(frames_dir, exist_ok=True)
                        
                        for i in range(len(self.animation_frames)):
                            # Update the figure with this frame
                            self.update_animation_frame(i)
                            
                            # Save the frame
                            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                            self.animation_fig.savefig(frame_path, dpi=self.animation_dpi)
                            
                        print(f"Saved {len(self.animation_frames)} individual frames to {frames_dir}")
                        
                        # Try to compile the frames into a video
                        self.animation_frames_dir = frames_dir
                        return self.try_compile_video_from_frames()
                    except Exception as e2:
                        print(f"Frame saving error: {e2}")
                        print("Could not save animation or frames.")
                        return None
            else:
                plt.show()
                return anim
                
        except Exception as e:
            print(f"Animation creation error: {e}")
            
            # As a fallback, at least save the final state
            try:
                if len(self.animation_frames) > 0:
                    # Update to final frame
                    self.update_animation_frame(len(self.animation_frames) - 1)
                    
                    # Save final frame
                    final_image_path = f"{os.path.splitext(self.animation_save_path)[0]}_final.png"
                    self.animation_fig.savefig(final_image_path, dpi=self.animation_dpi)
                    print(f"Saved final state image to {final_image_path}")
            except Exception:
                pass
                
            return None
    
    # Override the run_simulation method to capture frames
    original_run_simulation = simulator_class.run_simulation
    
    def animated_run_simulation(self, num_iterations=100000, batch_size=1000, 
                              use_parallel=True, pixels_per_move=20, 
                              capture_animation=False, animation_frames=100):
        """
        Run simulation with animation capture support
        
        Parameters:
        - Same as original run_simulation
        - capture_animation: Whether to capture frames for animation
        - animation_frames: Number of frames to capture
        """
        # Initialize animation if requested
        if capture_animation:
            if not hasattr(self, 'animation_fig'):
                self.setup_animation(frames=animation_frames)
            
            # Now we know the number of iterations, so we can calculate the frame interval
            self.animation_frame_interval = max(1, num_iterations // min(animation_frames, 150))  
            # Cap at 150 frames max to avoid memory issues
            print(f"Will capture animation frame every {self.animation_frame_interval} iterations")
            
            # Capture initial state
            self.capture_animation_frame()
        
        # Initialize the phase for iteration 0
        self.update_phase(0, num_iterations)
        
        if use_parallel and self.num_cpus > 1:
            print(f"Running simulation using {self.num_cpus} CPU cores in parallel with {pixels_per_move} pixels per move")
            iterations_completed = 0
            progress_bar = tqdm(total=num_iterations)
            
            while iterations_completed < num_iterations:
                # Determine current batch size
                current_batch_size = min(batch_size, num_iterations - iterations_completed)
                
                # Run batch with multiple pixels per move for early phases
                if self.phase <= 2:
                    accepted = self._process_batch_multi(batch_size=current_batch_size, pixels_per_move=pixels_per_move)
                else:
                    # For later phases focusing on compactness, use parallel processing with single pixel moves
                    accepted = self.run_batch_parallel(batch_size=current_batch_size)
                
                # Update iteration count
                iterations_completed += current_batch_size
                
                # Update phase
                self.update_phase(iterations_completed, num_iterations)
                
                # Update progress bar
                progress_bar.update(current_batch_size)
                
                # Capture animation frame if needed
                if capture_animation and iterations_completed % self.animation_frame_interval < current_batch_size:
                    self.capture_animation_frame()
                
                # Report progress more frequently
                if iterations_completed % 1000 == 0:
                    # Calculate statistics
                    self.calculate_all_district_stats()
                    
                    # Get population information
                    pop_mean = np.mean(self.district_stats['population'])
                    pop_max = np.max(self.district_stats['population'])
                    pop_min = np.min(self.district_stats['population'])
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
            
            # Clean up
            if hasattr(self, 'pool') and self.pool:
                self.pool.close()
                self.pool.join()
                self.pool = None
        else:
            # Implement single-threaded version with multi-pixel moves
            print(f"Running simulation in single-threaded mode with {pixels_per_move} pixels per move")
            progress_bar = tqdm(total=num_iterations)
            
            for i in range(0, num_iterations, pixels_per_move):
                # For early phases focusing on population balancing, use multi-pixel moves
                if self.phase <= 2:
                    self._process_batch_multi(batch_size=1, pixels_per_move=pixels_per_move)
                else:
                    # For later phases, use single pixel moves
                    for _ in range(pixels_per_move):
                        self.run_iteration()
                
                # Update iteration count
                current_iteration = min(i + pixels_per_move, num_iterations)
                
                # Update phase
                self.update_phase(current_iteration, num_iterations)
                
                # Capture animation frame if needed
                if capture_animation and current_iteration % self.animation_frame_interval < pixels_per_move:
                    self.capture_animation_frame()
                
                # Update progress every 1000 iterations
                if current_iteration % 1000 < pixels_per_move:
                    progress_bar.update(min(1000, current_iteration - progress_bar.n))
                    
                    # Report detailed status every 5000 iterations
                    if current_iteration % 5000 < pixels_per_move or current_iteration == num_iterations:
                        # Calculate statistics
                        self.calculate_all_district_stats()
                        
                        # Get population information
                        pop_mean = np.mean(self.district_stats['population'])
                        pop_max = np.max(self.district_stats['population'])
                        pop_min = np.min(self.district_stats['population'])
                        pop_imbalance = (pop_max - pop_min) / pop_mean
                        
                        # Get district counts
                        red_districts = sum(1 for d in range(self.num_districts) 
                                    if self.district_stats['red_votes'][d] > 
                                        self.district_stats['blue_votes'][d])
                        blue_districts = self.num_districts - red_districts
                        
                        # Output status
                        current_score = self.score_map()
                        print(f"\nIteration {current_iteration}/{num_iterations}, Score: {current_score:.2f}")
                        print(f"Population imbalance: {pop_imbalance:.2%}, Min: {pop_min:.0f}, Max: {pop_max:.0f}")
                        print(f"Districts: {red_districts} Red, {blue_districts} Blue")
                        print(f"Temperature: {self.temperature:.4f}, Phase: {self.phase}")
            
            progress_bar.close()
        
        # Capture final frame
        if capture_animation:
            self.capture_animation_frame()
        
        # Calculate final statistics
        self.calculate_all_district_stats()
        print("Simulation complete!")
        
        # Create animation if requested
        if capture_animation:
            return self.create_animation()
    
    # Now add all the new methods to the simulator class
    simulator_class.setup_animation = setup_animation
    simulator_class.capture_animation_frame = capture_animation_frame
    simulator_class.update_animation_frame = update_animation_frame
    simulator_class.try_compile_video_from_frames = try_compile_video_from_frames
    simulator_class.create_animation = create_animation
    simulator_class.run_simulation = animated_run_simulation
    
    return simulator_class