"""
Occupancy Grid Mapping (OGM) module for warehouse robot simulation.
Implements discrete grid mapping with frontier-based exploration.
"""

DEBUG = True  # Enable/disable debugging output

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[OGM DEBUG] {message}")

# Cell states
UNKNOWN = -1
FREE = 0
OCCUPIED = 1
GOAL = 2


class OccupancyGridMap:
    """
    Occupancy Grid Map for mapping the warehouse environment.
    Uses discrete states: UNKNOWN (-1), FREE (0), OCCUPIED (1), GOAL (2)
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
        # Initialize grid with UNKNOWN state
        self.grid = [[UNKNOWN for _ in range(width)] for _ in range(height)]
        # Track obstacles and goals
        self.obstacles = set()  # Set of (x, y) tuples
        self.goals = set()  # Set of (x, y) tuples
        # Track explored cells
        self.explored_cells = set()  # Set of (x, y) tuples
        self.loading_dock = None
        self.discharge_dock = None
        
        debug_print(f"Initialized OGM with dimensions {width}x{height}")
    
    def mark_obstacle(self, x, y):
        """Mark a cell as occupied (obstacle). Prevents duplicates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            if (x, y) not in self.obstacles:
                self.grid[y][x] = OCCUPIED
                self.obstacles.add((x, y))
                debug_print(f"Marked obstacle at ({x}, {y})")
    
    def mark_free(self, x, y):
        """Mark a cell as free space."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = FREE
            # Remove from obstacles if it was there
            self.obstacles.discard((x, y))
            debug_print(f"Marked free space at ({x}, {y})")
    
    def mark_goal(self, x, y):
        """Mark a cell as a goal location. Prevents duplicates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            if (x, y) not in self.goals:
                self.grid[y][x] = GOAL
                self.goals.add((x, y))
                debug_print(f"Marked goal at ({x}, {y})")
    
    def mark_explored(self, x, y):
        """Mark a cell as explored (visited by robot)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.explored_cells.add((x, y))
            # Mark as free if not already marked as goal
            if self.grid[y][x] != GOAL:
                self.mark_free(x, y)
    
    def is_explored(self, x, y):
        """Check if a cell has been explored."""
        return (x, y) in self.explored_cells
    
    def mark_loading_dock(self, x, y):
        """Mark the loading dock location."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.loading_dock = (x, y)
            self.mark_free(x, y)
            debug_print(f"Marked loading dock at ({x}, {y})")
    
    def mark_discharge_dock(self, x, y):
        """Mark the discharge dock location."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.discharge_dock = (x, y)
            self.mark_free(x, y)
            debug_print(f"Marked discharge dock at ({x}, {y})")
    
    def is_obstacle(self, x, y):
        """Check if a cell is an obstacle."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] == OCCUPIED
        return True  # Out of bounds is considered obstacle
    
    def is_free(self, x, y):
        """Check if a cell is free space."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] == FREE or self.grid[y][x] == GOAL
        return False
    
    def is_unknown(self, x, y):
        """Check if a cell is unknown."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] == UNKNOWN
        return False
    
    def get_cell_state(self, x, y):
        """Get the state of a cell."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return OCCUPIED  # Out of bounds is considered obstacle
    
    def update_from_grid_ray_cast(self, robot_rc, warehouse, R=8):
        """
        Update OGM from grid ray casting.
        Casts rays in 4 directions (N, E, S, W) and updates cells along the path.
        
        Args:
            robot_rc: Robot position (row, col) in grid coordinates
            warehouse: Warehouse object to check obstacles
            R: Ray range in grid cells
        """
        r0, c0 = robot_rc
        dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # N, S, E, W
        
        for dr, dc in dirs4:
            r, c = r0, c0
            for k in range(1, R + 1):
                r += dr
                c += dc
                
                # Check bounds
                if not (0 <= r < self.height and 0 <= c < self.width):
                    break
                
                # Check if hit obstacle in warehouse
                if warehouse.is_blocked(c, r):
                    # Mark hit cell as occupied
                    self.mark_obstacle(c, r)
                    break
                else:
                    # Mark traversed cell as free
                    self.mark_free(c, r)
        
        # Also check immediate neighbors for goals and docks
        directions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]  # Current, N, S, W, E
        for dr, dc in directions:
            r, c = r0 + dr, c0 + dc
            if 0 <= r < self.height and 0 <= c < self.width:
                # Check for goal
                for goal in warehouse.goals:
                    if goal[0] == c and goal[1] == r:
                        self.mark_goal(c, r)
                        break
                
                # Check for loading dock
                if warehouse.loading_dock and warehouse.loading_dock[0] == c and warehouse.loading_dock[1] == r:
                    self.mark_loading_dock(c, r)
                
                # Check for discharge dock
                if warehouse.discharge_dock and warehouse.discharge_dock[0] == c and warehouse.discharge_dock[1] == r:
                    self.mark_discharge_dock(c, r)
        
        debug_print(f"Updated OGM from grid ray cast at robot position ({c0}, {r0})")
    
    def get_map_summary(self):
        """Get a summary of the mapped environment."""
        obstacle_count = sum(1 for row in self.grid for cell in row if cell == OCCUPIED)
        free_count = sum(1 for row in self.grid for cell in row if cell == FREE)
        unknown_count = sum(1 for row in self.grid for cell in row if cell == UNKNOWN)
        goal_count = sum(1 for row in self.grid for cell in row if cell == GOAL)
        
        return {
            'obstacles': obstacle_count,
            'free': free_count,
            'unknown': unknown_count,
            'goals': goal_count,
            'loading_dock': self.loading_dock,
            'discharge_dock': self.discharge_dock
        }
