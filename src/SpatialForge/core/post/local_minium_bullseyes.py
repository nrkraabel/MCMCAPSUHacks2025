import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import math

class MCMCAnimation:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle('Markov Chain Monte Carlo Annealing with Multiple Local Minima', fontsize=16)
        
        self.targets = [
            {"x": 0, "y": 0, "radius": 5, "value": 0},
            {"x": -50, "y": -10, "radius": 3, "value": 1.5},
            {"x": -38, "y": -28, "radius": 3, "value": 2},
            {"x": -23, "y": -45, "radius": 3.5, "value": 2.5}
        ]
        
        self.grid_resolution = 100
        self.score_grid = self.create_score_landscape()
        
        self.x = -50
        self.y = -50
        self.temperature = 1
        self.cooling_rate = 0.999
        
        self.max_trail_length = 200
        self.trail_x = [self.x]
        self.trail_y = [self.y]
        
        self.step_count = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
        
        self.setup_plot()

    def create_score_landscape(self):
        """Create a grid of scores across the landscape to visualize the energy function"""
        x = np.linspace(-100, 100, self.grid_resolution)
        y = np.linspace(-100, 100, self.grid_resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                Z[i, j] = self.calculate_score(X[i, j], Y[i, j])
                
        return {"X": X, "Y": Y, "Z": Z}

    def setup_plot(self):
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        
        contour = self.ax.contourf(
            self.score_grid["X"], 
            self.score_grid["Y"], 
            self.score_grid["Z"], 
            levels=20, 
            cmap='viridis_r',
            alpha=0.6
        )
        self.fig.colorbar(contour, label='Score (lower is better)')
        
        self.create_bullseyes()
        
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        self.point, = self.ax.plot([self.x], [self.y], 'bo', markersize=10, label='Current Position')
        
        
        self.trail, = self.ax.plot(self.trail_x, self.trail_y, 'b-', alpha=0.5, linewidth=1.5, label='Path')
        
        self.temp_text = self.ax.text(0.02, 0.97, '', transform=self.ax.transAxes, fontsize=10)
        self.dist_text = self.ax.text(0.02, 0.93, '', transform=self.ax.transAxes, fontsize=10)
        self.step_text = self.ax.text(0.02, 0.89, '', transform=self.ax.transAxes, fontsize=10)
        self.accept_text = self.ax.text(0.02, 0.85, '', transform=self.ax.transAxes, fontsize=10)
        
        self.ax.legend(loc='upper right')

    def create_bullseyes(self):
        for target in self.targets:
            ring_count = 3
            for i in range(ring_count, 0, -1):
                radius = target["radius"] * (i + 0.5)
                if target["value"] == 0:
                    color = 'white' if i % 2 == 0 else 'red'
                else:
                    color = 'white' if i % 2 == 0 else 'orange'
                
                ring = patches.Circle(
                    (target["x"], target["y"]), 
                    radius, 
                    fill=True, 
                    color=color, 
                    zorder=i, 
                    alpha=0.7
                )
                self.ax.add_patch(ring)
            
            center_color = 'red' if target["value"] == 0 else 'orange'
            center = patches.Circle(
                (target["x"], target["y"]), 
                target["radius"], 
                fill=True, 
                color=center_color, 
                zorder=ring_count+1
            )
            self.ax.add_patch(center)
            
            label = "Global Minimum" if target["value"] == 0 else f"Local Min ({target['value']})"
            self.ax.annotate(
                label,
                (target["x"], target["y"] + target["radius"] * 4),
                ha='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
            )

    def calculate_score(self, x, y):
        """Score function based on distance to targets modified by their values"""
        scores = []
        for target in self.targets:
            distance = math.sqrt((x - target["x"])**2 + (y - target["y"])**2)
            
            if distance < target["radius"]:
                target_score = target["value"]
            else:
                dist_factor = max(0, (distance - target["radius"])**0.5)
                target_score = target["value"] + dist_factor
                
            scores.append(target_score)
        
        return min(scores)

    def update_frame(self, frame):
        x_move = (np.random.random() - 0.5) * 15
        y_move = (np.random.random() - 0.5) * 15
        
        new_x = self.x + x_move
        new_y = self.y + y_move
        
        new_x = max(-100, min(100, new_x))
        new_y = max(-100, min(100, new_y))
        
        current_score = self.calculate_score(self.x, self.y)
        new_score = self.calculate_score(new_x, new_y)
        
        accept_move = False
        
        if new_score <= current_score:
            accept_move = True
        else:
            delta = new_score - current_score
            acceptance_probability = math.exp(-delta / self.temperature)
            accept_move = np.random.random() < acceptance_probability
        
        if accept_move:
            self.x = new_x
            self.y = new_y
            self.accepted_moves += 1
            
            self.trail_x.append(self.x)
            self.trail_y.append(self.y)
            
            if len(self.trail_x) > self.max_trail_length:
                self.trail_x.pop(0)
                self.trail_y.pop(0)
                
            current_score = self.calculate_score(self.x, self.y)

        else:
            self.rejected_moves += 1
        
        self.temperature = max(0.1, self.temperature * self.cooling_rate)
        
        self.step_count += 1
        
        self.point.set_data([self.x], [self.y])
        self.trail.set_data(self.trail_x, self.trail_y)
        
        self.temp_text.set_text(f'Temperature: {self.temperature:.2f}')
        self.dist_text.set_text(f'Current score: {self.calculate_score(self.x, self.y):.2f}')
        self.step_text.set_text(f'Steps: {self.step_count}')
        
        acceptance_rate = (self.accepted_moves / self.step_count) * 100 if self.step_count > 0 else 0
        self.accept_text.set_text(f'Acceptance: {acceptance_rate:.1f}%')
        
        return self.point, self.trail, self.temp_text, self.dist_text, self.step_text, self.accept_text

    def run_animation(self, frames=1000, interval=30, save_path=None):
        self.animation = FuncAnimation(
            self.fig, self.update_frame, frames=frames, interval=interval, blit=True)
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            extension = save_path.split('.')[-1].lower()
            
            if extension == 'gif':
                self.animation.save(
                    save_path, 
                    writer='pillow', 
                    fps=30,
                    dpi=100
                )
            elif extension in ['mp4', 'mov', 'avi']:
                try:
                    from matplotlib.animation import FFMpegWriter
                    writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
                    self.animation.save(save_path, writer=writer)
                except (ImportError, ValueError):
                    # Fallback to other writers if ffmpeg is not available
                    try:
                        from matplotlib.animation import MovieWriter
                        self.animation.save(save_path, fps=30, dpi=100)
                    except Exception as e:
                        print(f"Error saving video: {e}")
                        print("Try installing ffmpeg for better video export.")
            else:
                self.animation.save(save_path, fps=30, dpi=100)
                
            print(f"Animation saved to {save_path}")
        
        plt.show()

if __name__ == "__main__":
    animation = MCMCAnimation()
    animation.run_animation(frames=1000, interval=30, save_path="mcmc_animation.gif")