import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import math

class MCMCAnimation:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle('Markov Chain Monte Carlo Annealing', fontsize=16)
        
        self.target_x = 0
        self.target_y = 0
        self.target_radius = 5
        
        self.x = 50
        self.y = 50
        self.temperature = 50 
        self.cooling_rate = 0.98 
        
        self.max_trail_length = 100
        self.trail_x = [self.x]
        self.trail_y = [self.y]
        
        self.step_count = 0
        self.accepted_moves = 0
        self.rejected_moves = 0
        
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        self.create_bullseye()
        
        self.point, = self.ax.plot([self.x], [self.y], 'bo', markersize=10, label='Current Position')
        
        self.trail, = self.ax.plot(self.trail_x, self.trail_y, 'b-', alpha=0.5, linewidth=1.5, label='Path')
        
        self.temp_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, fontsize=10)
        self.dist_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes, fontsize=10)
        self.step_text = self.ax.text(0.02, 0.85, '', transform=self.ax.transAxes, fontsize=10)
        self.accept_text = self.ax.text(0.02, 0.80, '', transform=self.ax.transAxes, fontsize=10)
        
        self.ax.legend(loc='upper right')

    def create_bullseye(self):
        ring_count = 5
        for i in range(ring_count, 0, -1):
            radius = self.target_radius * (i + 1)
            color = 'white' if i % 2 == 0 else 'red'
            ring = patches.Circle((self.target_x, self.target_y), radius, 
                                 fill=True, color=color, zorder=i, alpha=0.7)
            self.ax.add_patch(ring)
        
        center = patches.Circle((self.target_x, self.target_y), self.target_radius, 
                               fill=True, color='red', zorder=ring_count+1)
        self.ax.add_patch(center)

    def calculate_score(self, x, y):
        """Score function - distance to target (lower is better)"""
        return math.sqrt((x - self.target_x)**2 + (y - self.target_y)**2)

    def update_frame(self, frame):
        x_move = (np.random.random() - 0.5)*10
        y_move = (np.random.random() - 0.5)*10
        
        new_x = self.x + x_move
        new_y = self.y + y_move
        
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
        else:
            self.rejected_moves += 1
        
        self.temperature = max(0.1, self.temperature * self.cooling_rate)
        
        self.step_count += 1
        
        self.point.set_data([self.x], [self.y])
        self.trail.set_data(self.trail_x, self.trail_y)
        
        self.temp_text.set_text(f'Temperature: {self.temperature:.2f}')
        self.dist_text.set_text(f'Distance to target: {self.calculate_score(self.x, self.y):.2f}')
        self.step_text.set_text(f'Steps: {self.step_count}')
        
        acceptance_rate = (self.accepted_moves / self.step_count) * 100 if self.step_count > 0 else 0
        self.accept_text.set_text(f'Acceptance: {acceptance_rate:.1f}%')
        
        return self.point, self.trail, self.temp_text, self.dist_text, self.step_text, self.accept_text

    def run_animation(self, frames=500, interval=50):
        self.animation = FuncAnimation(
            self.fig, self.update_frame, frames=frames, interval=interval, blit=True)
        plt.show()

if __name__ == "__main__":
    animation = MCMCAnimation()
    animation.run_animation(frames=500, interval=50)