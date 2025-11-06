"""
Dynamic obstacles module for the warehouse robot simulation.
Implements moving obstacles (workers and forklifts) that move randomly.
"""

import random
import pygame
from constants import WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT, GRID_SIZE, debug_log


class DynamicObstacle:
    """Base class for dynamic obstacles (workers and forklifts)."""
    
    def __init__(self, x, y, obstacle_type='worker', warehouse=None):
        """
        Initialize a dynamic obstacle.
        
        Args:
            x: Starting x coordinate (grid cell)
            y: Starting y coordinate (grid cell)
            obstacle_type: Type of obstacle ('worker' or 'forklift')
            warehouse: Warehouse object for collision checking
        """
        self.x = float(x)
        self.y = float(y)
        self.obstacle_type = obstacle_type
        self.warehouse = warehouse
        
        # Movement properties
        self.movement_speed = 1  # Grid cells per move
        self.last_move_time = 0
        self.move_cooldown = 100  # milliseconds between moves (slower than robot)
        
        # Random movement state
        self.current_direction = None  # (dx, dy) tuple
        self.steps_in_direction = 0
        self.max_steps_in_direction = random.randint(3, 8)  # Move in same direction for 3-8 steps
        
        # Choose initial random direction
        self.choose_new_direction()
        
        debug_log(f"Created {obstacle_type} at ({int(x)}, {int(y)})")
    
    def choose_new_direction(self):
        """Choose a new random direction (up, down, left, right)."""
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        self.current_direction = random.choice(directions)
        self.steps_in_direction = 0
        self.max_steps_in_direction = random.randint(3, 8)
    
    def can_move_to(self, new_x, new_y):
        """Check if obstacle can move to a new position."""
        # Check bounds
        if new_x < 0 or new_x >= WAREHOUSE_WIDTH or new_y < 0 or new_y >= WAREHOUSE_HEIGHT:
            return False
        
        # Check warehouse static obstacles
        if self.warehouse and self.warehouse.is_blocked(int(new_x), int(new_y)):
            return False
        
        return True
    
    def update(self, current_time, other_obstacles=None):
        """
        Update obstacle position based on random movement.
        
        Args:
            current_time: Current time in milliseconds
            other_obstacles: List of other dynamic obstacles to avoid collisions with
        """
        # Check cooldown
        if current_time - self.last_move_time < self.move_cooldown:
            return
        
        # Check if we should change direction
        if self.steps_in_direction >= self.max_steps_in_direction:
            self.choose_new_direction()
        
        # Calculate new position
        dx, dy = self.current_direction
        new_x = self.x + dx * self.movement_speed
        new_y = self.y + dy * self.movement_speed
        
        # Check if we can move to new position
        if self.can_move_to(new_x, new_y):
            # Check collision with other dynamic obstacles
            collision = False
            if other_obstacles:
                for other in other_obstacles:
                    if other is not self:
                        # Check if new position would overlap with another obstacle
                        if abs(new_x - other.x) < 0.5 and abs(new_y - other.y) < 0.5:
                            collision = True
                            break
            
            if not collision:
                # Move to new position
                self.x = new_x
                self.y = new_y
                self.last_move_time = current_time
                self.steps_in_direction += 1
            else:
                # Collision detected - choose new direction
                self.choose_new_direction()
        else:
            # Can't move in current direction - choose new direction
            self.choose_new_direction()
    
    def draw(self, surface):
        """Draw the dynamic obstacle."""
        x_pixel = int(self.x * GRID_SIZE)
        y_pixel = int(self.y * GRID_SIZE)
        center_x = x_pixel + GRID_SIZE // 2
        center_y = y_pixel + GRID_SIZE // 2
        
        if self.obstacle_type == 'worker':
            # Draw worker as a person icon (circle with stick figure)
            # Body (circle)
            pygame.draw.circle(surface, (100, 150, 255), (center_x, center_y), GRID_SIZE // 3)
            # Head (smaller circle on top)
            pygame.draw.circle(surface, (255, 200, 150), (center_x, center_y - GRID_SIZE // 6), GRID_SIZE // 6)
            # Direction indicator (small line)
            dx, dy = self.current_direction if self.current_direction else (0, -1)
            if dx != 0 or dy != 0:
                indicator_x = center_x + dx * (GRID_SIZE // 4)
                indicator_y = center_y + dy * (GRID_SIZE // 4)
                pygame.draw.line(surface, (0, 0, 0), (center_x, center_y), (indicator_x, indicator_y), 2)
        elif self.obstacle_type == 'forklift':
            # Draw forklift as a rectangle with forks
            # Main body (rectangle)
            rect_size = GRID_SIZE // 2
            pygame.draw.rect(surface, (255, 100, 0), 
                           (center_x - rect_size // 2, center_y - rect_size // 2, 
                            rect_size, rect_size))
            # Forks (two lines at front)
            dx, dy = self.current_direction if self.current_direction else (0, -1)
            if dx != 0 or dy != 0:
                fork_length = GRID_SIZE // 3
                fork_x = center_x + dx * (rect_size // 2)
                fork_y = center_y + dy * (rect_size // 2)
                # Two parallel forks
                if dx != 0:  # Moving horizontally
                    pygame.draw.line(surface, (200, 200, 200), 
                                   (fork_x, fork_y - 5), 
                                   (fork_x + dx * fork_length, fork_y - 5), 3)
                    pygame.draw.line(surface, (200, 200, 200), 
                                   (fork_x, fork_y + 5), 
                                   (fork_x + dx * fork_length, fork_y + 5), 3)
                else:  # Moving vertically
                    pygame.draw.line(surface, (200, 200, 200), 
                                   (fork_x - 5, fork_y), 
                                   (fork_x - 5, fork_y + dy * fork_length), 3)
                    pygame.draw.line(surface, (200, 200, 200), 
                                   (fork_x + 5, fork_y), 
                                   (fork_x + 5, fork_y + dy * fork_length), 3)
            # Direction indicator
            if dx != 0 or dy != 0:
                indicator_x = center_x + dx * (GRID_SIZE // 4)
                indicator_y = center_y + dy * (GRID_SIZE // 4)
                pygame.draw.line(surface, (0, 0, 0), (center_x, center_y), (indicator_x, indicator_y), 2)


class Worker(DynamicObstacle):
    """Worker dynamic obstacle."""
    
    def __init__(self, x, y, warehouse=None):
        super().__init__(x, y, obstacle_type='worker', warehouse=warehouse)


class Forklift(DynamicObstacle):
    """Forklift dynamic obstacle."""
    
    def __init__(self, x, y, warehouse=None):
        super().__init__(x, y, obstacle_type='forklift', warehouse=warehouse)

