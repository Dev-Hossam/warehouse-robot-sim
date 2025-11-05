"""
Grid ray casting module for warehouse robot simulation.
Implements discrete grid-based ray casting in 4 directions.
"""

DEBUG = True  # Enable/disable debugging output

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[LIDAR DEBUG] {message}")


class LidarSensor:
    """
    Grid-based ray casting sensor that casts rays in 4 directions (N, E, S, W).
    Updates occupancy grid map with observed cells.
    """
    
    def __init__(self, max_range=8):
        """
        Initialize the grid ray casting sensor.
        
        Args:
            max_range: Maximum range of rays (in grid cells)
        """
        self.max_range = max_range
        # 4 directions: N, S, E, W (as (dr, dc) tuples)
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # N, S, E, W
        
        debug_print(f"Grid ray casting sensor initialized: max_range={max_range}")
    
    def update_ogm_with_rays(self, robot_rc, ogm, warehouse):
        """
        Cast rays in 4 directions and update OGM.
        
        Args:
            robot_rc: Robot position (row, col) in grid coordinates
            ogm: OccupancyGridMap to update
            warehouse: Warehouse object to check obstacles
        """
        r0, c0 = robot_rc
        R = self.max_range
        
        for dr, dc in self.directions:
            r, c = r0, c0
            for k in range(1, R + 1):
                r += dr
                c += dc
                
                # Check bounds
                if not (0 <= r < ogm.height and 0 <= c < ogm.width):
                    break
                
                # Check if hit obstacle in warehouse (ground truth)
                if warehouse.is_blocked(c, r):
                    # Mark hit cell as occupied
                    ogm.mark_obstacle(c, r)
                    break
                else:
                    # Mark traversed cell as free
                    ogm.mark_free(c, r)
        
        # Also check immediate neighbors for goals and docks
        directions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]  # Current, N, S, W, E
        for dr, dc in directions:
            r, c = r0 + dr, c0 + dc
            if 0 <= r < ogm.height and 0 <= c < ogm.width:
                # Check for goal
                for goal in warehouse.goals:
                    if goal[0] == c and goal[1] == r:
                        ogm.mark_goal(c, r)
                        break
                
                # Check for loading dock
                if warehouse.loading_dock and warehouse.loading_dock[0] == c and warehouse.loading_dock[1] == r:
                    ogm.mark_loading_dock(c, r)
                
                # Check for discharge dock
                if warehouse.discharge_dock and warehouse.discharge_dock[0] == c and warehouse.discharge_dock[1] == r:
                    ogm.mark_discharge_dock(c, r)
        
        debug_print(f"Updated OGM with grid ray cast at robot position ({c0}, {r0})")
