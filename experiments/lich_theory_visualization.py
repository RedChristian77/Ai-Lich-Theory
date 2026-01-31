"""
LICH THEORY VISUALIZATION - Twitter Ready
==========================================

Animated GIF showing basin formation in real-time.
Points diverge while the ball stays round.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os

# Create figure
fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

# Remove axes
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Title
title = ax.text(0, 1.3, 'Basin Formation in Attention Space', 
                fontsize=16, color='white', ha='center', fontweight='bold')

subtitle = ax.text(0, -1.3, 'Same shape. Different internal geography.', 
                   fontsize=11, color='#888888', ha='center', style='italic')

# The "ball" - softmax constraint (sum = 1)
circle = Circle((0, 0), 1.0, fill=False, color='#4fc3f7', linewidth=3)
ax.add_patch(circle)

# Two points that will diverge
point_a, = ax.plot([], [], 'o', color='#ff6b6b', markersize=15, label='Pattern A')
point_b, = ax.plot([], [], 'o', color='#4ecdc4', markersize=15, label='Pattern B')

# Trail lines
trail_a, = ax.plot([], [], '-', color='#ff6b6b', alpha=0.3, linewidth=2)
trail_b, = ax.plot([], [], '-', color='#4ecdc4', alpha=0.3, linewidth=2)

# Distance text
dist_text = ax.text(0, -0.1, '', fontsize=14, color='white', ha='center')

# Iteration text
iter_text = ax.text(0, 1.1, '', fontsize=12, color='#888888', ha='center')

# Legend
ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#333333',
          labelcolor='white', fontsize=10)

# Animation data
n_frames = 100

# Starting positions (close together)
start_a = np.array([0.1, 0.3])
start_b = np.array([0.15, 0.32])

# Ending positions (diverged, still on unit circle constraint)
end_a = np.array([-0.5, 0.6])
end_b = np.array([0.6, -0.4])

# Normalize to stay "inside" the ball but show movement
def normalize_position(pos, max_radius=0.85):
    norm = np.linalg.norm(pos)
    if norm > max_radius:
        return pos / norm * max_radius
    return pos

# Store trails
trail_a_x, trail_a_y = [], []
trail_b_x, trail_b_y = [], []

def init():
    point_a.set_data([], [])
    point_b.set_data([], [])
    trail_a.set_data([], [])
    trail_b.set_data([], [])
    dist_text.set_text('')
    iter_text.set_text('')
    return point_a, point_b, trail_a, trail_b, dist_text, iter_text

def animate(frame):
    global trail_a_x, trail_a_y, trail_b_x, trail_b_y
    
    # Progress (0 to 1)
    t = frame / n_frames
    
    # Smooth easing
    t_smooth = t * t * (3 - 2 * t)  # Smoothstep
    
    # Interpolate positions
    pos_a = start_a + (end_a - start_a) * t_smooth
    pos_b = start_b + (end_b - start_b) * t_smooth
    
    # Add some organic movement
    wobble = 0.02 * np.sin(frame * 0.3)
    pos_a = pos_a + np.array([wobble, -wobble])
    pos_b = pos_b + np.array([-wobble, wobble])
    
    # Normalize
    pos_a = normalize_position(pos_a)
    pos_b = normalize_position(pos_b)
    
    # Update points
    point_a.set_data([pos_a[0]], [pos_a[1]])
    point_b.set_data([pos_b[0]], [pos_b[1]])
    
    # Update trails
    trail_a_x.append(pos_a[0])
    trail_a_y.append(pos_a[1])
    trail_b_x.append(pos_b[0])
    trail_b_y.append(pos_b[1])
    
    # Keep trail length limited
    max_trail = 30
    if len(trail_a_x) > max_trail:
        trail_a_x = trail_a_x[-max_trail:]
        trail_a_y = trail_a_y[-max_trail:]
        trail_b_x = trail_b_x[-max_trail:]
        trail_b_y = trail_b_y[-max_trail:]
    
    trail_a.set_data(trail_a_x, trail_a_y)
    trail_b.set_data(trail_b_x, trail_b_y)
    
    # Calculate distance
    distance = np.linalg.norm(pos_a - pos_b)
    dist_text.set_text(f'Distance: {distance:.3f}')
    
    # Iteration
    iter_text.set_text(f'Training iteration: {frame}')
    
    return point_a, point_b, trail_a, trail_b, dist_text, iter_text

# Create animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=n_frames, interval=50, blit=True)

# Save as GIF
output_path = '/mnt/user-data/outputs/basin_formation.gif'
anim.save(output_path, writer='pillow', fps=20, dpi=100)

print(f"Saved to: {output_path}")

# Also save a static "before/after" image
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1a1a2e')

for ax in [ax1, ax2]:
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    circle = Circle((0, 0), 1.0, fill=False, color='#4fc3f7', linewidth=3)
    ax.add_patch(circle)

# Before
ax1.plot([0.1], [0.3], 'o', color='#ff6b6b', markersize=20)
ax1.plot([0.15], [0.32], 'o', color='#4ecdc4', markersize=20)
ax1.set_title('Before Training', color='white', fontsize=14, fontweight='bold')
ax1.text(0, -1.2, 'Distance: 0.054', color='white', ha='center', fontsize=12)

# After
ax2.plot([-0.5], [0.6], 'o', color='#ff6b6b', markersize=20)
ax2.plot([0.6], [-0.4], 'o', color='#4ecdc4', markersize=20)
ax2.set_title('After Training', color='white', fontsize=14, fontweight='bold')
ax2.text(0, -1.2, 'Distance: 1.414', color='white', ha='center', fontsize=12)

fig2.suptitle('Lich Theory: Basins Form Inside the Attention Space', 
              color='white', fontsize=16, fontweight='bold', y=0.98)

static_path = '/mnt/user-data/outputs/basin_before_after.png'
fig2.savefig(static_path, facecolor='#1a1a2e', edgecolor='none', 
             bbox_inches='tight', dpi=150)

print(f"Saved static image to: {static_path}")

plt.close('all')
print("Done!")
