"""
Occupancy Grid Mapping (OGM) module for warehouse robot simulation.
Implements mapping algorithm to identify obstacles and goals in the grid.
"""

import math

DEBUG = True  # Enable/disable debugging output

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[OGM DEBUG] {message}")


class OccupancyGridMap:
    """
    Occupancy Grid Map for mapping the warehouse environment.
    Uses a probabilistic approach with log-odds for SLAM.
    """
    
    def __init__(self, width, height):
        """
        Initialize the occupancy grid map.
        
        Args:
            width: Grid width in cells
            height: Grid height in cells
        """
        self.width = width
        self.height = height
        # Initialize grid with unknown probability (0.5)
        # 0.0 = free, 1.0 = occupied, 0.5 = unknown
        # Store as probabilities for backwards compatibility
        self.grid = [[0.5 for _ in range(width)] for _ in range(height)]
        # Log-odds representation for probabilistic updates
        # log_odds = log(p / (1-p)), where p is probability
        # Initialize with log_odds(0.5) = 0
        self.log_odds_grid = [[0.0 for _ in range(width)] for _ in range(height)]
        # Track obstacles and goals (use sets to prevent duplicates)
        self.obstacles = set()  # Set of (x, y) tuples
        self.goals = set()  # Set of (x, y) tuples (changed from list to prevent duplicates)
        # Track explored cells - cells that have been visited/explored
        self.explored_cells = set()  # Set of (x, y) tuples
        self.loading_dock = None
        self.discharge_dock = None
        
        # Inverse sensor model parameters
        self.l_occ = 0.8  # Log-odds update for occupied cells
        self.l_free = -0.4  # Log-odds update for free cells
        self.l_prior = 0.0  # Prior log-odds (unknown = 0)
        
        debug_print(f"Initialized OGM with dimensions {width}x{height}")
    
    def probability_to_log_odds(self, probability):
        """
        Convert probability to log-odds.
        
        Args:
            probability: Probability value (0.0-1.0)
        
        Returns:
            float: Log-odds value
        """
        if probability <= 0:
            return -float('inf')
        elif probability >= 1:
            return float('inf')
        elif probability == 0.5:
            return 0.0
        else:
            return math.log(probability / (1.0 - probability))
    
    def log_odds_to_probability(self, log_odds):
        """
        Convert log-odds to probability.
        
        Args:
            log_odds: Log-odds value
        
        Returns:
            float: Probability value (0.0-1.0)
        """
        if log_odds == float('inf'):
            return 1.0
        elif log_odds == -float('inf'):
            return 0.0
        else:
            return 1.0 / (1.0 + math.exp(-log_odds))
    
    def update_cell(self, x, y, probability):
        """
        Update a cell's occupancy probability.
        
        Args:
            x: X coordinate
            y: Y coordinate
            probability: Occupancy probability (0.0-1.0)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            old_prob = self.grid[y][x]
            self.grid[y][x] = max(0.0, min(1.0, probability))
            # Update log-odds accordingly
            self.log_odds_grid[y][x] = self.probability_to_log_odds(self.grid[y][x])
            debug_print(f"Updated cell ({x}, {y}): {old_prob:.2f} -> {self.grid[y][x]:.2f}")
    
    def update_cell_log_odds(self, x, y, log_odds_update):
        """
        Update a cell using log-odds (for probabilistic updates).
        
        Args:
            x: X coordinate
            y: Y coordinate
            log_odds_update: Log-odds update to add (can be negative for free space)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            # Update log-odds
            self.log_odds_grid[y][x] += log_odds_update
            # Clamp log-odds to reasonable range to avoid overflow
            self.log_odds_grid[y][x] = max(-10.0, min(10.0, self.log_odds_grid[y][x]))
            # Convert back to probability
            self.grid[y][x] = self.log_odds_to_probability(self.log_odds_grid[y][x])
            debug_print(f"Updated cell ({x}, {y}) log-odds: {log_odds_update:.3f}, "
                       f"new prob: {self.grid[y][x]:.3f}")
    
    def mark_obstacle(self, x, y):
        """Mark a cell as occupied (obstacle). Prevents duplicates."""
        if (x, y) not in self.obstacles:
            self.update_cell(x, y, 1.0)
            self.obstacles.add((x, y))
            debug_print(f"Marked obstacle at ({x}, {y})")
    
    def mark_free(self, x, y):
        """Mark a cell as free space."""
        self.update_cell(x, y, 0.0)
        debug_print(f"Marked free space at ({x}, {y})")
    
    def mark_goal(self, x, y):
        """Mark a cell as a goal location. Prevents duplicates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            if (x, y) not in self.goals:
                self.goals.add((x, y))
                debug_print(f"Marked goal at ({x}, {y})")
    
    def mark_explored(self, x, y):
        """Mark a cell as explored (visited by robot)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.explored_cells.add((x, y))
            # Mark as free - if it was previously marked as obstacle, correct it
            # This ensures OGM is corrected when we successfully visit a cell
            if (x, y) in self.obstacles:
                # Remove from obstacles set if we successfully visited it
                self.obstacles.discard((x, y))
            self.mark_free(x, y)
    
    def is_explored(self, x, y):
        """Check if a cell has been explored."""
        return (x, y) in self.explored_cells
    
    def mark_loading_dock(self, x, y):
        """Mark the loading dock location."""
        self.loading_dock = (x, y)
        self.mark_free(x, y)
        debug_print(f"Marked loading dock at ({x}, {y})")
    
    def mark_discharge_dock(self, x, y):
        """Mark the discharge dock location."""
        self.discharge_dock = (x, y)
        self.mark_free(x, y)
        debug_print(f"Marked discharge dock at ({x}, {y})")
    
    def is_obstacle(self, x, y):
        """Check if a cell is an obstacle (probability > 0.7)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] > 0.7
        return True  # Out of bounds is considered obstacle
    
    def is_free(self, x, y):
        """Check if a cell is free space (probability < 0.3)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] < 0.3
        return False
    
    def get_cell_probability(self, x, y):
        """Get the occupancy probability of a cell."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return 1.0  # Out of bounds is considered obstacle
    
    def update_from_lidar_scan(self, scan_results, robot_pose):
        """
        Update OGM from LIDAR scan using inverse sensor model.
        Uses log-odds for probabilistic updates.
        
        Args:
            scan_results: Dictionary with 'ray_hits' containing ray hit information
            robot_pose: Tuple (x, y, theta) of robot's estimated pose
        """
        robot_x, robot_y, robot_theta = robot_pose
        
        # Process each ray hit
        if 'ray_hits' in scan_results:
            for ray_hit in scan_results['ray_hits']:
                hit_x = ray_hit['x']
                hit_y = ray_hit['y']
                distance = ray_hit['distance']
                angle = ray_hit['angle']
                hit_type = ray_hit['type']
                
                # Calculate ray direction in robot's frame
                angle_rad = math.radians(angle)
                dx = math.cos(angle_rad)
                dy = -math.sin(angle_rad)  # Negative because y increases downward
                
                # Update cells along the ray path
                for step in range(1, int(distance) + 1):
                    check_x = int(robot_x + dx * step)
                    check_y = int(robot_y + dy * step)
                    
                    # Skip out of bounds
                    if not (0 <= check_x < self.width and 0 <= check_y < self.height):
                        continue
                    
                    # If this is the last cell (hit), mark as occupied
                    if step == int(distance) and hit_type in ['obstacle', 'boundary']:
                        self.update_cell_log_odds(check_x, check_y, self.l_occ)
                    else:
                        # Cells along the ray path before hit are free
                        self.update_cell_log_odds(check_x, check_y, self.l_free)
        
        # Also process immediate free cells from scan
        if 'free_cells' in scan_results:
            for free_x, free_y in scan_results['free_cells']:
                if 0 <= free_x < self.width and 0 <= free_y < self.height:
                    self.update_cell_log_odds(free_x, free_y, self.l_free)
        
        # Process detected obstacles (prevent duplicates)
        if 'obstacles' in scan_results:
            for obs_x, obs_y in scan_results['obstacles']:
                if 0 <= obs_x < self.width and 0 <= obs_y < self.height:
                    if (obs_x, obs_y) not in self.obstacles:
                        self.update_cell_log_odds(obs_x, obs_y, self.l_occ)
                        self.obstacles.add((obs_x, obs_y))
        
        debug_print(f"Updated OGM from LIDAR scan at robot pose ({robot_x:.2f}, {robot_y:.2f}, {robot_theta:.2f}°)")
    
    def map_from_warehouse(self, warehouse):
        """
        Map the warehouse environment using provided warehouse data.
        This is a simplified mapping - in a real scenario, the robot would
        use sensors to discover the environment.
        
        Args:
            warehouse: Warehouse object with obstacles and goals
        """
        debug_print("Starting OGM mapping from warehouse data...")
        
        # Map all obstacles
        debug_print(f"Mapping {len(warehouse.obstacles)} obstacles...")
        for x, y in warehouse.obstacles:
            self.mark_obstacle(x, y)
        
        # Map goals (prevent duplicates)
        debug_print(f"Mapping {len(warehouse.goals)} goals...")
        for goal in warehouse.goals:
            goal_x, goal_y = goal[0], goal[1]
            if (goal_x, goal_y) not in self.goals:
                self.mark_goal(goal_x, goal_y)
                # Goals are on free space
                self.mark_free(goal_x, goal_y)
        
        # Map loading dock
        if warehouse.loading_dock:
            self.mark_loading_dock(warehouse.loading_dock[0], warehouse.loading_dock[1])
        
        # Map discharge dock
        if warehouse.discharge_dock:
            self.mark_discharge_dock(warehouse.discharge_dock[0], warehouse.discharge_dock[1])
        
        # Don't mark cells as explored/free during initialization - they start unexplored
        # Cells will be marked as explored when robot visits them
        # Only mark obstacles and goals, keep other cells as unknown (0.5 probability)
        
        debug_print(f"OGM mapping complete!")
        debug_print(f"  Obstacles: {len(self.obstacles)}")
        debug_print(f"  Goals: {len(self.goals)}")
        debug_print(f"  Cells start as unexplored (unknown) - will be marked as explored when visited")
        debug_print(f"  Loading dock: {self.loading_dock}")
        debug_print(f"  Discharge dock: {self.discharge_dock}")
    
    def get_map_summary(self):
        """Get a summary of the mapped environment."""
        obstacle_count = sum(1 for row in self.grid for cell in row if cell > 0.7)
        free_count = sum(1 for row in self.grid for cell in row if cell < 0.3)
        unknown_count = sum(1 for row in self.grid for cell in row if 0.3 <= cell <= 0.7)
        
        return {
            'obstacles': obstacle_count,
            'free': free_count,
            'unknown': unknown_count,
            'goals': len(self.goals),
            'loading_dock': self.loading_dock,
            'discharge_dock': self.discharge_dock
        }
