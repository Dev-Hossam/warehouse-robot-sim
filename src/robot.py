"""
Robot module for the warehouse robot simulation.
Implements frontier-based exploration with iSAM localization.
"""

import math
import pygame
import numpy as np
from constants import WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT, GRID_SIZE, BLUE, RED, BLACK, debug_log
from ogm import OccupancyGridMap, UNKNOWN, FREE, OCCUPIED, GOAL
from lidar import LidarSensor
from isam import ISAM
from astar import astar, plan_multi_goal_path

DEBUG = True

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[ROBOT DEBUG] {message}")


class Robot:
    """
    Robot class with frontier-based exploration and iSAM localization.
    """
    
    def __init__(self, x, y, ogm=None, warehouse=None):
        """
        Initialize the robot.
        
        Args:
            x: Starting x coordinate
            y: Starting y coordinate
            ogm: Occupancy Grid Map for mapping
            warehouse: Warehouse object (for validation)
        """
        # Validate starting position
        if warehouse and warehouse.is_blocked(int(x), int(y)):
            debug_log(f"WARNING: Robot starting position ({x}, {y}) is on obstacle! Adjusting...")
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    test_x, test_y = int(x) + dx, int(y) + dy
                    if not warehouse.is_blocked(test_x, test_y):
                        x, y = test_x, test_y
                        break
                if x != int(x) or y != int(y):
                    break
        
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        
        # Movement properties
        self.movement_speed = 1  # Grid cells per move
        self.rotation_angle = 0  # Current rotation angle in degrees
        self.target_rotation = 0  # Target rotation angle
        self.rotation_speed = 180  # Degrees per rotation (increased from 90°)
        
        # Robot state
        self.current_goal = None
        self.has_cargo = False
        self.last_move_time = 0
        self.move_cooldown = 25  # milliseconds between moves (increased speed from 50ms to 25ms)
        self.last_action_time = 0
        self.action_cooldown = 200  # milliseconds between actions (reduced from 300ms)
        self.score = 0
        
        # Mapping and exploration
        if ogm is None:
            from constants import WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT
            self.ogm = OccupancyGridMap(WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT)
        else:
            self.ogm = ogm
        self.mapping_complete = False
        self.is_mapping = False
        self.warehouse = warehouse
        
        # Initialize grid ray casting sensor
        self.lidar = LidarSensor(max_range=8)
        
        # Initialize iSAM system
        self.isam = ISAM(initial_x=x, initial_y=y, initial_theta=0)
        
        # Track pose trajectory
        self.pose_trajectory = [(x, y)]
        self.loop_closure_detected = False
        self.last_node_pos = np.array([x, y])
        self.last_node_angle = 0
        
        # Exploration state
        self.exploration_mode = "EXPLORE"  # EXPLORE, RETURN_TO_START, or DELIVER_GOALS
        
        # DFS coverage: stack-based backtracking
        self.visited = set()  # Set of (x, y) visited cells
        self.stack = []  # Stack for backtracking: [(x, y), ...]
        
        # Cached exploration completeness check
        self.has_unknown_adjacent_cached = True  # Start with True, will be updated incrementally
        self.last_unknown_check_time = 0  # Track when we last checked (in frames)
        self.unknown_check_interval = 60  # Re-check every 60 frames (~1 second at 60 FPS)
        
        # Return to start after exploration
        self.return_path = []  # Path back to start position
        self.return_path_index = 0
        
        # Goal delivery state
        self.delivery_path = []  # Current path to goal or discharge dock
        self.delivery_path_index = 0
        self.delivery_mode = "PICKUP"  # PICKUP or DROPOFF
        self.goals_to_deliver = []  # Remaining goals to deliver (in priority order)
        
        # Path caching to avoid redundant A* calculations
        self.path_cache = {}  # Cache: {(start, target): path}
        self.path_cache_max_size = 100  # Maximum number of cached paths
        
        debug_log(f"Robot initialized at ({x}, {y}) with DFS coverage exploration")
    
    def rotate_to(self, target_angle):
        """Set target rotation angle."""
        self.target_rotation = target_angle % 360
    
    def rotate_towards(self, dx, dy):
        """Rotate robot towards a direction."""
        if dx == 1 and dy == 0:
            self.target_rotation = 0  # Right
        elif dx == -1 and dy == 0:
            self.target_rotation = 180  # Left
        elif dx == 0 and dy == -1:
            self.target_rotation = 90  # Up
        elif dx == 0 and dy == 1:
            self.target_rotation = 270  # Down
    
    def update_rotation(self, current_time):
        """Update robot rotation towards target angle."""
        angle_diff = (self.target_rotation - self.rotation_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        
        if abs(angle_diff) > 0.1:
            if abs(angle_diff) <= self.rotation_speed:
                self.rotation_angle = self.target_rotation
            else:
                if angle_diff > 0:
                    self.rotation_angle = (self.rotation_angle + self.rotation_speed) % 360
                else:
                    self.rotation_angle = (self.rotation_angle - self.rotation_speed) % 360
    
    def update_with_sensor(self):
        """Update OGM using grid ray casting."""
        if self.warehouse and self.ogm:
            robot_rc = (int(self.y), int(self.x))  # (row, col)
            self.lidar.update_ogm_with_rays(robot_rc, self.ogm, self.warehouse)
            # Mark current cell as explored
            current_x, current_y = int(self.x), int(self.y)
            self.ogm.mark_explored(current_x, current_y)
            
            # Update cached unknown check incrementally
            if hasattr(self, 'has_unknown_adjacent_cached'):
                if not self.check_unknown_adjacent_incremental(current_x, current_y):
                    # If this cell and its neighbors don't have unknown adjacent, 
                    # we might still have unknown elsewhere, but we'll check when stack is empty
                    # For now, keep the cached value as is (we'll do full check when needed)
                    pass
            
            # Check if we're at a goal position and mark it
            for goal in self.warehouse.goals:
                goal_x, goal_y = goal[0], goal[1]
                if current_x == goal_x and current_y == goal_y:
                    self.ogm.mark_goal(current_x, current_y)
                    debug_log(f"Reached goal at ({current_x}, {current_y})")
    
    def can_move_to(self, new_x, new_y):
        """Check if robot can move to a new position."""
        # Check bounds
        if new_x < 0 or new_x >= WAREHOUSE_WIDTH or new_y < 0 or new_y >= WAREHOUSE_HEIGHT:
            return False
        
        # Check warehouse obstacle (ground truth)
        if self.warehouse and self.warehouse.is_blocked(int(new_x), int(new_y)):
            return False
        
        # Check OGM obstacle
        if self.ogm and self.ogm.is_obstacle(int(new_x), int(new_y)):
            return False
        
        return True
    
    def move_to(self, new_x, new_y, current_time):
        """Move robot to a new position."""
        if not self.can_move_to(new_x, new_y):
            return False
        
        # Calculate movement delta
        dx = new_x - self.x
        dy = new_y - self.y
        
        # Move robot
        self.x = new_x
        self.y = new_y
        self.last_move_time = current_time
        
        # Update iSAM pose estimate
        dtheta = (self.target_rotation - self.rotation_angle) % 360
        if dtheta > 180:
            dtheta -= 360
        self.isam.update_pose(dx, dy, dtheta)
        
        # Add node to pose graph if moved far enough
        current_pos = np.array([self.x, self.y])
        distance_moved = np.linalg.norm(current_pos - self.last_node_pos)
        if distance_moved >= 2.0:  # Add node every 2 grid cells
            self.isam.add_node(current_pos.copy(), self.rotation_angle)
            self.last_node_pos = current_pos.copy()
            self.last_node_angle = self.rotation_angle
            
            # Detect loop closure
            loop_node = self.isam.detect_loop_closure(current_pos, self.rotation_angle)
            if loop_node is not None:
                self.loop_closure_detected = True
                if self.isam.previous_node is not None:
                    self.isam.pose_graph.add_edge(self.isam.previous_node, loop_node)
            else:
                self.loop_closure_detected = False
        
        # Add to trajectory
        self.pose_trajectory.append((self.x, self.y))
        if len(self.pose_trajectory) > 1000:
            self.pose_trajectory.pop(0)
        
        # Update sensor
        self.update_with_sensor()
        
        return True
    
    def neighbors4(self, x, y):
        """Get 4-neighbors of a cell in fixed order (N, E, S, W)."""
        # Fixed order: N, E, S, W (up, right, down, left)
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < WAREHOUSE_WIDTH and 0 <= ny < WAREHOUSE_HEIGHT:
                yield (nx, ny)
    
    def has_unvisited_free_neighbor(self, x, y):
        """Check if cell has an unvisited FREE neighbor."""
        for nx, ny in self.neighbors4(x, y):
            if (nx, ny) not in self.visited:
                if self.can_move_to(nx, ny):
                    return True
        return False
    
    def get_unvisited_free_neighbor(self, x, y):
        """Get first unvisited FREE neighbor in fixed order."""
        for nx, ny in self.neighbors4(x, y):
            if (nx, ny) not in self.visited:
                if self.can_move_to(nx, ny):
                    return (nx, ny)
        return None
    
    def check_unknown_adjacent_incremental(self, x, y):
        """Check if a newly explored cell (x, y) or its neighbors have unknown adjacent cells."""
        # Check if this cell or its neighbors are FREE/GOAL and have UNKNOWN neighbors
        cell_state = self.ogm.get_cell_state(x, y)
        if cell_state == FREE or cell_state == GOAL:
            # Check if this cell has UNKNOWN neighbors
            for nx, ny in self.neighbors4(x, y):
                if self.ogm.is_unknown(nx, ny):
                    return True
            # Check neighbors of this cell
            for nx, ny in self.neighbors4(x, y):
                neighbor_state = self.ogm.get_cell_state(nx, ny)
                if neighbor_state == FREE or neighbor_state == GOAL:
                    for nnx, nny in self.neighbors4(nx, ny):
                        if self.ogm.is_unknown(nnx, nny):
                            return True
        return False
    
    def check_unknown_adjacent_full(self):
        """Full grid scan to check if any FREE/GOAL cell has an UNKNOWN neighbor."""
        # Only do full scan when absolutely necessary (when stack is empty)
        for y in range(WAREHOUSE_HEIGHT):
            for x in range(WAREHOUSE_WIDTH):
                cell_state = self.ogm.get_cell_state(x, y)
                if cell_state == FREE or cell_state == GOAL:
                    # Check if any neighbor is UNKNOWN
                    for nx, ny in self.neighbors4(x, y):
                        if self.ogm.is_unknown(nx, ny):
                            return True
        return False
    
    def validate_exploration_complete(self):
        """Validate that exploration is complete (no UNKNOWN cells adjacent to FREE/GOAL cells)."""
        has_unknown = self.check_unknown_adjacent_full()
        self.has_unknown_adjacent_cached = has_unknown
        if has_unknown:
            debug_log("WARNING: Exploration not complete - UNKNOWN cells still adjacent to FREE/GOAL cells")
            return False
        debug_log("Exploration validation passed - no UNKNOWN cells adjacent to FREE/GOAL cells")
        return True
    
    def astar_cached(self, start, target, allow_goals=True):
        """
        A* pathfinding with caching to avoid redundant calculations.
        
        Args:
            start: (x, y) starting position
            target: (x, y) target position
            allow_goals: If True, GOAL cells are traversable
            
        Returns:
            list: Path as list of (x, y) tuples, or None if unreachable
        """
        # Check cache first
        cache_key = (start, target, allow_goals)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Plan path using A* module
        path = astar(start, target, self.ogm, allow_goals=allow_goals)
        
        # Cache the result (limit cache size)
        if len(self.path_cache) >= self.path_cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.path_cache))
            del self.path_cache[oldest_key]
        
        self.path_cache[cache_key] = path
        return path
    
    def astar(self, start, target):
        """
        A* pathfinding from start to target using the astar module.
        
        Args:
            start: (row, col) starting position
            target: (row, col) target position
            
        Returns:
            list: Path as list of (row, col) tuples, or None if unreachable
        """
        # Convert (row, col) to (x, y)
        sr, sc = start
        tr, tc = target
        start_xy = (sc, sr)  # (x, y)
        target_xy = (tc, tr)  # (x, y)
        
        # Use cached A* method
        path = self.astar_cached(start_xy, target_xy, allow_goals=True)
        
        if path:
            # Convert path from (x, y) to (row, col)
            path_rc = [(y, x) for x, y in path]
            return path_rc
        
        return None
    
    def plan_return_to_start(self):
        """Plan path back to starting position after exploration is complete."""
        current_pos = (int(self.x), int(self.y))  # (x, y)
        start_pos = (int(self.start_x), int(self.start_y))  # (x, y)
        
        # Ensure start position is marked as FREE in OGM
        if self.ogm:
            self.ogm.mark_free(int(self.start_x), int(self.start_y))
            self.ogm.mark_explored(int(self.start_x), int(self.start_y))
        
        # Ensure current position is marked as FREE in OGM
        if self.ogm:
            self.ogm.mark_free(int(self.x), int(self.y))
            self.ogm.mark_explored(int(self.x), int(self.y))
        
        debug_log(f"Planning return path from ({current_pos[0]}, {current_pos[1]}) to ({start_pos[0]}, {start_pos[1]})")
        
        # Plan path using cached A* method
        path = self.astar_cached(current_pos, start_pos, allow_goals=True)
        
        if path:
            if len(path) > 1:
                self.return_path = path[1:]  # Skip first cell (current position)
            else:
                # Already at start or path is just one cell
                self.return_path = []
            self.return_path_index = 0
            debug_log(f"Planned return path to start: {len(self.return_path)} steps (path length: {len(path)})")
        else:
            debug_log(f"Warning: Could not plan path from ({current_pos[0]}, {current_pos[1]}) to ({start_pos[0]}, {start_pos[1]})")
            # Check OGM states for debugging
            if self.ogm:
                current_state = self.ogm.get_cell_state(current_pos[0], current_pos[1])
                start_state = self.ogm.get_cell_state(start_pos[0], start_pos[1])
                debug_log(f"  Current position state: {current_state}, Start position state: {start_state}")
            self.return_path = []
            self.return_path_index = 0
    
    def plan_goal_delivery(self):
        """Initialize goal delivery state with goals in priority order."""
        if not self.warehouse or not self.warehouse.goals:
            debug_log("No goals to deliver")
            return
        
        # Get goals in priority order (they're already in order)
        self.goals_to_deliver = list(self.warehouse.goals)
        self.delivery_mode = "PICKUP"
        self.delivery_path = []
        self.delivery_path_index = 0
        debug_log(f"Initialized goal delivery: {len(self.goals_to_deliver)} goals to deliver")

    def explore_next(self, current_time):
        """Execute one step of DFS coverage exploration."""
        if not self.is_mapping:
            return False
        
        # Check cooldown
        if current_time - self.last_move_time < self.move_cooldown:
            return False
        
        # Mark current cell as visited
        current_pos = (int(self.x), int(self.y))
        if current_pos not in self.visited:
            self.visited.add(current_pos)
        
        if self.exploration_mode == "EXPLORE":
            # Check if there's an unvisited FREE neighbor
            next_pos = self.get_unvisited_free_neighbor(int(self.x), int(self.y))
            
            if next_pos:
                # Move to unvisited neighbor and push to stack
                nx, ny = next_pos
                # Push current position to stack BEFORE moving
                self.stack.append(current_pos)
                if self.move_to(nx, ny, current_time):
                    # Rotate towards direction
                    dx = nx - int(self.x)
                    dy = ny - int(self.y)
                    self.rotate_towards(dx, dy)
                    # Only log significant stack size changes (every 10 cells or when stack size changes significantly)
                    if len(self.stack) % 10 == 0 or len(self.stack) == 1:
                        debug_log(f"DFS: Exploring - stack size: {len(self.stack)}, visited: {len(self.visited)}")
                    return True
                else:
                    # Move failed, pop from stack and mark as visited to avoid retry
                    if self.stack and self.stack[-1] == current_pos:
                        self.stack.pop()
                    self.visited.add(next_pos)
                    return False
            else:
                # No unvisited neighbor - backtrack
                if self.stack:
                    # Pop from stack and move back
                    prev_pos = self.stack.pop()
                    px, py = prev_pos
                    if self.move_to(px, py, current_time):
                        # Rotate towards direction
                        dx = px - int(self.x)
                        dy = py - int(self.y)
                        self.rotate_towards(dx, dy)
                        # Only log significant backtrack events (every 10 stack pops or when stack becomes small)
                        if len(self.stack) % 10 == 0 or len(self.stack) < 5:
                            debug_log(f"DFS: Backtracking - stack size: {len(self.stack)}, visited: {len(self.visited)}")
                        return True
                    else:
                        # Backtrack failed, try next position in stack
                        # Only log failures, not every attempt
                        if len(self.stack) % 10 == 0:
                            debug_log(f"DFS: Backtrack failed, stack size: {len(self.stack)}")
                        return False
                else:
                    # Stack empty - check if exploration is complete (100% parsed)
                    # Use cached value and only re-check every 60 frames (~1 second) to reduce expensive scans
                    current_frame = current_time // (1000 // 60)  # Approximate frame count
                    if current_frame - self.last_unknown_check_time >= self.unknown_check_interval:
                        has_unknown_adjacent = self.check_unknown_adjacent_full()
                        self.has_unknown_adjacent_cached = has_unknown_adjacent
                        self.last_unknown_check_time = current_frame
                    else:
                        # Use cached value
                        has_unknown_adjacent = self.has_unknown_adjacent_cached
                    
                    if not has_unknown_adjacent:
                        # No UNKNOWN cells adjacent to FREE/GOAL cells - 100% exploration complete!
                        # Validate exploration completeness before transitioning
                        if self.validate_exploration_complete():
                            debug_log("=" * 50)
                            debug_log("EXPLORATION COMPLETE! 100% map parsed!")
                            debug_log(f"Visited: {len(self.visited)} cells")
                            debug_log(f"Goals discovered: {len(self.ogm.goals)}")
                            debug_log("Breaking out and returning to start position...")
                            debug_log("=" * 50)
                            
                            # Break out: switch to return mode
                            self.exploration_mode = "RETURN_TO_START"
                            self.mapping_complete = True
                            # Plan path back to start
                            self.plan_return_to_start()
                            return True
                        else:
                            # Validation failed - should not happen, but handle gracefully
                            debug_log("ERROR: Exploration validation failed despite check passing")
                            return False
                    else:
                        # Still have unknown cells but no path to them
                        debug_log("DFS: Stack empty but unknown cells remain - may be unreachable")
                        return False
        
        elif self.exploration_mode == "RETURN_TO_START":
            # After 100% exploration, return to starting position
            # Check if already at start position
            if abs(self.x - self.start_x) < 0.5 and abs(self.y - self.start_y) < 0.5:
                # Already at start - transition to delivery phase
                debug_log("=" * 50)
                debug_log(f"RETURNED TO START POSITION ({int(self.start_x)}, {int(self.start_y)})")
                debug_log("Exploration mission complete!")
                debug_log("Starting goal delivery phase...")
                debug_log("=" * 50)
                
                # Switch to goal delivery mode
                self.exploration_mode = "DELIVER_GOALS"
                self.delivery_mode = "PICKUP"
                # Plan goal delivery paths
                self.plan_goal_delivery()
                return True
            
            if not hasattr(self, 'return_path') or not self.return_path:
                # Plan return path if not already planned
                self.plan_return_to_start()
                return False
            
            # Execute return path
            if self.return_path_index < len(self.return_path):
                next_cell = self.return_path[self.return_path_index]
                nx, ny = next_cell
                
                if current_time - self.last_move_time >= self.move_cooldown:
                    if self.move_to(nx, ny, current_time):
                        self.return_path_index += 1
                        # Rotate towards direction
                        if self.return_path_index < len(self.return_path):
                            nx2, ny2 = self.return_path[self.return_path_index]
                            dx = nx2 - nx
                            dy = ny2 - ny
                            self.rotate_towards(dx, dy)
                        # Only log every 10 steps or when near completion
                        remaining = len(self.return_path) - self.return_path_index
                        if remaining % 10 == 0 or remaining < 5:
                            debug_log(f"Returning to start: {remaining} steps remaining")
                        return True
            else:
                # Reached start position
                if abs(self.x - self.start_x) < 0.5 and abs(self.y - self.start_y) < 0.5:
                    # Validate exploration completeness before starting delivery
                    if self.validate_exploration_complete():
                        debug_log("=" * 50)
                        debug_log(f"RETURNED TO START POSITION ({int(self.start_x)}, {int(self.start_y)})")
                        debug_log("Exploration mission complete!")
                        debug_log("Starting goal delivery phase...")
                        debug_log("=" * 50)
                        
                        # Switch to goal delivery mode
                        self.exploration_mode = "DELIVER_GOALS"
                        self.delivery_mode = "PICKUP"
                        # Plan goal delivery paths
                        self.plan_goal_delivery()
                        return True
                    else:
                        debug_log("ERROR: Cannot start delivery - exploration not complete")
                        return False
        
        elif self.exploration_mode == "DELIVER_GOALS":
            # Goal delivery phase: pick up goals and deliver to discharge dock
            if not self.delivery_path or self.delivery_path_index >= len(self.delivery_path):
                # Path complete or no path - check if we're at a goal or dock
                current_pos = (int(self.x), int(self.y))
                
                # Check if we're at a goal (pickup mode)
                if self.delivery_mode == "PICKUP" and not self.has_cargo:
                    # Always process goals in priority order (first goal in list)
                    if not self.goals_to_deliver:
                        # No more goals
                        debug_log("All goals delivered!")
                        self.is_mapping = False
                        return False
                    
                    # Get the first priority goal
                    next_goal = self.goals_to_deliver[0]
                    goal_x, goal_y = next_goal[0], next_goal[1]
                    
                    # Check if we're already at the first priority goal
                    if abs(self.x - goal_x) < 0.5 and abs(self.y - goal_y) < 0.5:
                        # Pick up cargo at first priority goal
                        self.has_cargo = True
                        self.current_goal = next_goal
                        self.delivery_mode = "DROPOFF"
                        debug_log(f"Picked up cargo at goal ({goal_x}, {goal_y}) - priority goal")
                        
                        # Plan path to discharge dock
                        discharge_dock = self.warehouse.discharge_dock
                        if discharge_dock:
                            path = self.astar_cached(current_pos, discharge_dock, allow_goals=True)
                            if path and len(path) > 1:
                                self.delivery_path = path[1:]
                                self.delivery_path_index = 0
                                debug_log(f"Planned path to discharge dock: {len(self.delivery_path)} steps")
                                return False
                            else:
                                debug_log(f"Warning: No path from goal to discharge dock")
                                return False
                        else:
                            debug_log(f"Warning: No discharge dock found")
                            return False
                    else:
                        # Not at goal yet - plan path to first priority goal
                        path = self.astar_cached(current_pos, next_goal, allow_goals=True)
                        if path and len(path) > 1:
                            self.delivery_path = path[1:]
                            self.delivery_path_index = 0
                            # Calculate priority number (1-based, first goal is priority 1)
                            # Priority = total initial goals - remaining goals + 1
                            total_initial_goals = len(self.warehouse.goals) + len(self.goals_to_deliver) if self.warehouse else len(self.goals_to_deliver)
                            priority = total_initial_goals - len(self.goals_to_deliver) + 1
                            debug_log(f"Planned path to goal {next_goal} (priority {priority}): {len(self.delivery_path)} steps")
                            return False
                        else:
                            debug_log(f"Warning: No path to goal {next_goal}, removing from list")
                            self.goals_to_deliver.remove(next_goal)
                            return False
                
                # Check if we're at discharge dock (dropoff mode)
                elif self.delivery_mode == "DROPOFF" and self.has_cargo:
                    discharge_dock = self.warehouse.discharge_dock
                    if discharge_dock:
                        dock_x, dock_y = discharge_dock[0], discharge_dock[1]
                        if abs(self.x - dock_x) < 0.5 and abs(self.y - dock_y) < 0.5:
                            # Drop cargo
                            self.has_cargo = False
                            if self.current_goal in self.warehouse.goals:
                                self.warehouse.goals.remove(self.current_goal)
                            if self.current_goal in self.goals_to_deliver:
                                self.goals_to_deliver.remove(self.current_goal)
                            self.score += 1
                            self.current_goal = None
                            self.delivery_mode = "PICKUP"
                            debug_log(f"Dropped cargo at discharge dock ({dock_x}, {dock_y}). Score: {self.score}")
                            
                            # Check if more goals to deliver
                            if self.goals_to_deliver:
                                # Plan path to next goal (first priority goal)
                                current_pos = (int(self.x), int(self.y))
                                next_goal = self.goals_to_deliver[0]
                                path = self.astar_cached(current_pos, next_goal, allow_goals=True)
                                if path and len(path) > 1:
                                    self.delivery_path = path[1:]
                                    self.delivery_path_index = 0
                                    total_initial_goals = len(self.warehouse.goals) + len(self.goals_to_deliver) if self.warehouse else len(self.goals_to_deliver)
                                    priority = total_initial_goals - len(self.goals_to_deliver) + 1
                                    debug_log(f"Planned path to next goal {next_goal} (priority {priority}): {len(self.delivery_path)} steps")
                                    return False
                                else:
                                    debug_log(f"Warning: No path to next goal {next_goal}, removing from list")
                                    self.goals_to_deliver.remove(next_goal)
                                    return False
                            else:
                                # All goals delivered
                                debug_log("=" * 50)
                                debug_log("ALL GOALS DELIVERED! Mission complete!")
                                debug_log(f"Final score: {self.score}")
                                debug_log("=" * 50)
                                self.is_mapping = False
                                return False
                
                # No more goals or path failed
                return False
            
            # Execute delivery path
            if self.delivery_path_index < len(self.delivery_path):
                next_cell = self.delivery_path[self.delivery_path_index]
                nx, ny = next_cell
                
                if current_time - self.last_move_time >= self.move_cooldown:
                    if self.move_to(nx, ny, current_time):
                        self.delivery_path_index += 1
                        # Rotate towards direction
                        if self.delivery_path_index < len(self.delivery_path):
                            nx2, ny2 = self.delivery_path[self.delivery_path_index]
                            dx = nx2 - nx
                            dy = ny2 - ny
                            self.rotate_towards(dx, dy)
                        # Only log every 10 steps or when near completion
                        remaining = len(self.delivery_path) - self.delivery_path_index
                        if remaining % 10 == 0 or remaining < 5:
                            debug_log(f"Delivery: {remaining} steps remaining")
                        return True
        
        return False
    
    def start_mapping(self, warehouse):
        """Start autonomous mapping phase."""
        self.warehouse = warehouse
        
        if self.ogm is None:
            from constants import WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT
            self.ogm = OccupancyGridMap(WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT)
        
        debug_log("=" * 50)
        debug_log("STARTING FRONTIER-BASED EXPLORATION")
        debug_log("=" * 50)
        
        # Reset iSAM
        self.isam.reset()
        self.isam.set_pose(self.x, self.y, 0)
        self.last_node_pos = np.array([self.x, self.y])
        self.last_node_angle = 0
        
        # Add initial node
        self.isam.add_node(self.last_node_pos.copy(), self.last_node_angle)
        
        self.is_mapping = True
        self.mapping_complete = False
        self.exploration_mode = "EXPLORE"
        self.pose_trajectory = [(self.x, self.y)]
        
        # Reset DFS state
        self.visited = set()
        self.stack = []
        self.return_path = []
        self.return_path_index = 0
        
        # Reset cached exploration completeness check
        self.has_unknown_adjacent_cached = True  # Start with True, will be updated incrementally
        self.last_unknown_check_time = 0
        
        # Reset delivery state
        self.delivery_path = []
        self.delivery_path_index = 0
        self.delivery_mode = "PICKUP"
        self.goals_to_deliver = []
        
        # Reset path cache
        self.path_cache = {}
        self.path_cache_max_size = 100
        
        # Mark starting position as explored
        self.update_with_sensor()
        start_pos = (int(self.x), int(self.y))
        self.visited.add(start_pos)
        self.stack.append(start_pos)  # Start with initial position in stack
        
        debug_log(f"Starting DFS coverage exploration from position ({int(self.x)}, {int(self.y)})")
    
    def finish_mapping(self):
        """Finish the mapping phase."""
        self.is_mapping = False
        self.mapping_complete = True
        debug_log("Mapping phase complete!")
    
    def handle_input(self, keys, warehouse, current_time):
        """Handle keyboard input for robot movement."""
        if current_time - self.last_move_time < self.move_cooldown:
            return
        
        new_x, new_y = self.x, self.y
        moved = False
        
        if keys[pygame.K_UP] and self.y > 0:
            new_y -= self.movement_speed
            moved = True
            self.rotate_to(90)
        elif keys[pygame.K_DOWN] and self.y < WAREHOUSE_HEIGHT - 1:
            new_y += self.movement_speed
            moved = True
            self.rotate_to(270)
        elif keys[pygame.K_LEFT] and self.x > 0:
            new_x -= self.movement_speed
            moved = True
            self.rotate_to(180)
        elif keys[pygame.K_RIGHT] and self.x < WAREHOUSE_WIDTH - 1:
            new_x += self.movement_speed
            moved = True
            self.rotate_to(0)
        
        if moved:
            self.move_to(new_x, new_y, current_time)
    
    def try_pickup(self, warehouse, current_time):
        """Try to pick up cargo at current goal location."""
        if current_time - self.last_action_time < self.action_cooldown:
            return False
        
        if self.has_cargo or not self.current_goal:
            return False
        
        goal_x, goal_y = self.current_goal
        if abs(self.x - goal_x) < 1.5 and abs(self.y - goal_y) < 1.5:
            self.has_cargo = True
            warehouse.goals.remove(self.current_goal)
            self.last_action_time = current_time
            debug_log(f"Picked up cargo at goal ({goal_x}, {goal_y})")
            return True
        
        return False
    
    def try_drop(self, warehouse, current_time):
        """Try to drop cargo at discharge dock."""
        if current_time - self.last_action_time < self.action_cooldown:
            return False
        
        if not self.has_cargo:
            return False
        
        dock_x, dock_y = warehouse.discharge_dock
        if abs(self.x - dock_x) < 1.5 and abs(self.y - dock_y) < 1.5:
            self.has_cargo = False
            self.score += 1
            self.last_action_time = current_time
            debug_log(f"Dropped cargo at discharge dock ({dock_x}, {dock_y}). Score: {self.score}")
            if warehouse.goals:
                self.current_goal = warehouse.goals[0]
            else:
                self.current_goal = None
            return True
        
        return False
    
    def draw(self, surface):
        """Draw the robot with rotation, estimated pose, and trajectory."""
        # Draw trajectory
        if len(self.pose_trajectory) > 1:
            trajectory_points = [(int(x * GRID_SIZE + GRID_SIZE // 2), 
                                 int(y * GRID_SIZE + GRID_SIZE // 2)) 
                                for x, y in self.pose_trajectory[-100:]]
            if len(trajectory_points) > 1:
                pygame.draw.lines(surface, (200, 200, 200), False, trajectory_points, 2)
        
        # Draw estimated pose from iSAM (green circle)
        estimated_pose = self.isam.get_estimated_pose()
        est_x_pixel = estimated_pose[0] * GRID_SIZE
        est_y_pixel = estimated_pose[1] * GRID_SIZE
        est_center_x = est_x_pixel + GRID_SIZE // 2
        est_center_y = est_y_pixel + GRID_SIZE // 2
        
        # Draw uncertainty ellipse
        uncertainty = self.isam.get_uncertainty()
        pos_uncertainty = uncertainty[0]
        ellipse_radius = int(pos_uncertainty * GRID_SIZE * 2)
        if ellipse_radius > 0:
            pygame.draw.ellipse(
                surface, (150, 255, 150),
                (est_center_x - ellipse_radius, est_center_y - ellipse_radius,
                 ellipse_radius * 2, ellipse_radius * 2), 1
            )
        
        # Draw estimated pose (green circle)
        pygame.draw.circle(surface, (0, 255, 0), (est_center_x, est_center_y), GRID_SIZE // 4)
        
        # Draw line from estimated to actual pose
        x_pixel = self.x * GRID_SIZE
        y_pixel = self.y * GRID_SIZE
        center_x = x_pixel + GRID_SIZE // 2
        center_y = y_pixel + GRID_SIZE // 2
        
        pygame.draw.line(
            surface, (255, 200, 0), (est_center_x, est_center_y), (center_x, center_y), 2
        )
        
        # Draw robot body (blue circle)
        pygame.draw.circle(surface, BLUE, (center_x, center_y), GRID_SIZE // 3)
        
        # Draw rotation indicator
        angle_rad = math.radians(self.rotation_angle)
        line_length = GRID_SIZE // 2
        end_x = center_x + math.cos(angle_rad) * line_length
        end_y = center_y - math.sin(angle_rad) * line_length
        pygame.draw.line(surface, BLACK, (center_x, center_y), (int(end_x), int(end_y)), 3)
        
        # Draw cargo indicator
        if self.has_cargo:
            pygame.draw.circle(surface, RED, (center_x, center_y), GRID_SIZE // 5)
        
        # Draw loop closure indicator
        if self.loop_closure_detected:
            estimated_pose = self.isam.get_estimated_pose()
            loop_x = int(estimated_pose[0] * GRID_SIZE + GRID_SIZE // 2)
            loop_y = int(estimated_pose[1] * GRID_SIZE + GRID_SIZE // 2)
            pygame.draw.circle(surface, (255, 0, 255), (loop_x, loop_y), GRID_SIZE // 2, 3)
