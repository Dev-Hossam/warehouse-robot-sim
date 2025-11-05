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
        self.rotation_speed = 90  # Degrees per rotation
        
        # Robot state
        self.current_goal = None
        self.has_cargo = False
        self.last_move_time = 0
        self.move_cooldown = 100  # milliseconds between moves
        self.last_action_time = 0
        self.action_cooldown = 300  # milliseconds between actions
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
        self.exploration_mode = "EXPLORE"  # EXPLORE, RETURN_TO_START, or PLAN_TO_GOALS
        
        # DFS coverage: stack-based backtracking
        self.visited = set()  # Set of (x, y) visited cells
        self.stack = []  # Stack for backtracking: [(x, y), ...]
        
        # Return to start after exploration
        self.return_path = []  # Path back to start position
        self.return_path_index = 0
        
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
    
    
    def astar(self, start, target):
        """
        A* pathfinding from start to target.
        
        Args:
            start: (row, col) starting position
            target: (row, col) target position
            
        Returns:
            list: Path as list of (row, col) tuples, or None if unreachable
        """
        if start == target:
            return [start]
        
        sr, sc = start
        tr, tc = target
        
        # Check if target is valid
        if not (0 <= tr < WAREHOUSE_HEIGHT and 0 <= tc < WAREHOUSE_WIDTH):
            return None
        if self.ogm.is_obstacle(tc, tr):
            return None
        
        # A* algorithm
        open_set = [(0, sr, sc)]
        came_from = {}
        g_score = {(sr, sc): 0}
        f_score = {(sr, sc): abs(tr - sr) + abs(tc - sc)}  # Manhattan heuristic
        
        while open_set:
            open_set.sort(key=lambda x: x[0])
            current_f, cr, cc = open_set.pop(0)
            
            if (cr, cc) == (tr, tc):
                # Reconstruct path
                path = []
                node = (cr, cc)
                while node is not None:
                    path.append(node)
                    node = came_from.get(node)
                path.reverse()
                return path
            
            # Check neighbors
            for nr, nc in self.neighbors4(cr, cc):
                if self.ogm.is_obstacle(nc, nr):
                    continue
                
                tentative_g = g_score[(cr, cc)] + 1
                
                if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                    came_from[(nr, nc)] = (cr, cc)
                    g_score[(nr, nc)] = tentative_g
                    h = abs(tr - nr) + abs(tc - nc)  # Manhattan heuristic
                    f = tentative_g + h
                    f_score[(nr, nc)] = f
                    
                    # Add to open set if not already there
                    if not any(nr == r and nc == c for _, r, c in open_set):
                        open_set.append((f, nr, nc))
        
        return None  # No path found
    
    def plan_return_to_start(self):
        """Plan path back to starting position after exploration is complete."""
        current_pos = (int(self.y), int(self.x))  # (row, col)
        start_pos = (int(self.start_y), int(self.start_x))  # (row, col)
        
        # Plan path using A*
        path = self.astar(current_pos, start_pos)
        
        if path and len(path) > 1:
            # Convert path from (row, col) to (x, y)
            self.return_path = []
            for r, c in path[1:]:  # Skip first cell (current position)
                self.return_path.append((c, r))  # Convert (row, col) to (x, y)
            self.return_path_index = 0
            debug_log(f"Planned return path to start: {len(self.return_path)} steps")
        else:
            debug_log("Warning: Could not plan path to start position")
            self.return_path = []
            self.return_path_index = 0

    def explore_next(self, current_time):
        """Execute one step of DFS coverage exploration."""
        if not self.is_mapping:
            return False
        
        # Check cooldown
        if current_time - self.last_move_time < self.move_cooldown:
            return False
        
        # Update sensor
        self.update_with_sensor()
        
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
                    debug_log(f"DFS: Moving to unvisited neighbor ({nx}, {ny}), stack size: {len(self.stack)}")
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
                        debug_log(f"DFS: Backtracking to ({px}, {py}), stack size: {len(self.stack)}")
                        return True
                    else:
                        # Backtrack failed, try next position in stack
                        debug_log(f"DFS: Backtrack failed to ({px}, {py}), trying next")
                        return False
                else:
                    # Stack empty - check if exploration is complete (100% parsed)
                    # Check if any FREE or GOAL cell has an UNKNOWN neighbor
                    has_unknown_adjacent = False
                    for y in range(WAREHOUSE_HEIGHT):
                        for x in range(WAREHOUSE_WIDTH):
                            cell_state = self.ogm.get_cell_state(x, y)
                            if cell_state == FREE or cell_state == GOAL:
                                # Check if any neighbor is UNKNOWN
                                for nx, ny in self.neighbors4(x, y):
                                    if self.ogm.is_unknown(nx, ny):
                                        has_unknown_adjacent = True
                                        break
                                if has_unknown_adjacent:
                                    break
                        if has_unknown_adjacent:
                            break
                    
                    if not has_unknown_adjacent:
                        # No UNKNOWN cells adjacent to FREE/GOAL cells - 100% exploration complete!
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
                        # Still have unknown cells but no path to them
                        debug_log("DFS: Stack empty but unknown cells remain - may be unreachable")
                        return False
        
        elif self.exploration_mode == "RETURN_TO_START":
            # After 100% exploration, return to starting position
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
                        debug_log(f"Returning to start: ({nx}, {ny}), {len(self.return_path) - self.return_path_index} steps remaining")
                        return True
            else:
                # Reached start position
                if abs(self.x - self.start_x) < 0.5 and abs(self.y - self.start_y) < 0.5:
                    debug_log("=" * 50)
                    debug_log(f"RETURNED TO START POSITION ({int(self.start_x)}, {int(self.start_y)})")
                    debug_log("Exploration mission complete!")
                    debug_log("=" * 50)
                    self.is_mapping = False
                    return False
        
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
