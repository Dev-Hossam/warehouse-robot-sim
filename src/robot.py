"""
Robot module for the warehouse robot simulation.
Includes rotation, movement, and autonomous exploration capabilities with SLAM.
"""

import math
import pygame
from constants import WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT, GRID_SIZE, BLUE, RED, BLACK, debug_log
from ogm import OccupancyGridMap
from lidar import LidarSensor
from odometry import Odometry
from slam import SLAMSystem

DEBUG = True

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[ROBOT DEBUG] {message}")


class Robot:
    """
    Robot class with rotation, movement, and autonomous exploration capabilities with SLAM.
    
    PHYSICS ENFORCEMENT:
    - All movement MUST go through move_to() method to ensure obstacle physics are enforced
    - Robot CANNOT move through obstacles at any time (during scanning, exploration, or manual control)
    - can_move_to() checks warehouse obstacles (ground truth), OGM obstacles, and LIDAR path clearance
    - Position is validated before and after movement to ensure robot never ends up on obstacles
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
        # Validate starting position is not on obstacle
        if warehouse and warehouse.is_blocked(int(x), int(y)):
            debug_log(f"WARNING: Robot starting position ({x}, {y}) is on obstacle! Adjusting...")
            # Find nearest free cell
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    test_x, test_y = int(x) + dx, int(y) + dy
                    if not warehouse.is_blocked(test_x, test_y):
                        x, y = test_x, test_y
                        debug_log(f"Adjusted starting position to ({x}, {y})")
                        break
                if x != int(x) or y != int(y):
                    break
        
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        
        # Movement and rotation properties
        self.movement_speed = 1  # Grid cells per move
        self.rotation_speed = 90  # Degrees per rotation (for rotation animation)
        self.rotation_angle = 0  # Current rotation angle in degrees (0 = right, 90 = up, 180 = left, 270 = down)
        self.target_rotation = 0  # Target rotation angle
        
        # Robot state
        self.current_goal = None
        self.has_cargo = False
        self.last_move_time = 0
        self.move_cooldown = 200  # milliseconds between moves
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
        self.exploration_queue = []  # Cells to explore (BFS queue)
        self.visited = set()  # Already visited cells
        self.queued_cells = set()  # Track cells already in queue to avoid duplicates
        self.is_mapping = False  # Currently mapping phase
        self.warehouse = warehouse  # Store warehouse reference for exploration
        
        # Initialize LIDAR sensor
        self.lidar = LidarSensor(max_range=10, num_rays=8)
        
        # Initialize SLAM components
        # Ground truth pose: self.x, self.y, self.rotation_angle (for visualization)
        # Estimated pose: tracked by SLAM system (for mapping decisions)
        self.odometry = Odometry(initial_x=x, initial_y=y, initial_theta=0)
        self.slam = SLAMSystem(initial_x=x, initial_y=y, initial_theta=0)
        
        # Track pose trajectory for visualization
        self.pose_trajectory = [(x, y)]
        self.loop_closure_detected = False
        self.loop_closure_position = None
        self.last_loop_closure_check = 0  # Track when we last checked for loop closure
        
        debug_log(f"Robot initialized at ({x}, {y}) with LIDAR sensor and SLAM system")
    
    def rotate_to(self, target_angle):
        """
        Set target rotation angle.
        
        Args:
            target_angle: Target angle in degrees (0 = right, 90 = up, 180 = left, 270 = down)
        """
        self.target_rotation = target_angle % 360
    
    def rotate_towards(self, dx, dy):
        """
        Rotate robot towards a direction.
        
        Args:
            dx: Delta x (-1, 0, or 1)
            dy: Delta y (-1, 0, or 1)
        """
        if dx == 1 and dy == 0:
            self.target_rotation = 0  # Right
        elif dx == -1 and dy == 0:
            self.target_rotation = 180  # Left
        elif dx == 0 and dy == -1:
            self.target_rotation = 90  # Up
        elif dx == 0 and dy == 1:
            self.target_rotation = 270  # Down
    
    def update_rotation(self, current_time):
        """
        Update robot rotation towards target angle.
        
        Args:
            current_time: Current game time
        """
        # Calculate angle difference
        angle_diff = (self.target_rotation - self.rotation_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        
        # Rotate towards target
        if abs(angle_diff) > 0.1:  # Small threshold to avoid jitter
            if abs(angle_diff) <= self.rotation_speed:
                self.rotation_angle = self.target_rotation
            else:
                if angle_diff > 0:
                    self.rotation_angle = (self.rotation_angle + self.rotation_speed) % 360
                else:
                    self.rotation_angle = (self.rotation_angle - self.rotation_speed) % 360
    
    def sense_surroundings(self, warehouse):
        """
        Use LIDAR sensor to detect obstacles, goals, and docks in surrounding cells.
        Updates OGM using SLAM with probabilistic updates and loop closure detection.
        
        Args:
            warehouse: Warehouse object to sense
        """
        # CRITICAL: Verify robot is not on an obstacle (should never happen, but safety check)
        if warehouse.is_blocked(int(self.x), int(self.y)):
            debug_log(f"PHYSICS ERROR: Robot is on obstacle at ({self.x}, {self.y}) during scanning!")
            # Try to find a safe adjacent cell and move there
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for dx, dy in directions:
                safe_x = int(self.x) + dx
                safe_y = int(self.y) + dy
                # Use validated movement check
                if not warehouse.is_blocked(safe_x, safe_y) and 0 <= safe_x < WAREHOUSE_WIDTH and 0 <= safe_y < WAREHOUSE_HEIGHT:
                    debug_log(f"Moving robot to safe position ({safe_x}, {safe_y})")
                    # Direct assignment for emergency correction (already validated)
                    self.x = safe_x
                    self.y = safe_y
                    break
        
        # Get estimated pose from SLAM
        estimated_pose = self.slam.get_estimated_pose()
        
        # Perform LIDAR scan from ground truth position (actual world)
        # The scan represents what the robot actually sees from its true position
        scan_results = self.lidar.scan(self.x, self.y, warehouse)
        
        # Use actual pose for OGM updates to ensure map accuracy
        # SLAM estimated pose is used for localization only, not for mapping
        # Update OGM using SLAM with inverse sensor model (use actual pose for mapping)
        if self.ogm is not None:
            actual_pose = (self.x, self.y, self.rotation_angle)
            self.ogm.update_from_lidar_scan(scan_results, actual_pose)
        
        # Update OGM with discovered obstacles (prevent duplicates)
        for obstacle_pos in scan_results['obstacles']:
            obs_x, obs_y = obstacle_pos
            if self.ogm and (obs_x, obs_y) not in self.ogm.obstacles:
                self.ogm.mark_obstacle(obs_x, obs_y)
                debug_log(f"LIDAR: NEW obstacle discovered at ({obs_x}, {obs_y})")
        
        # Update OGM with discovered goals (prevent duplicates)
        for goal_pos in scan_results['goals']:
            goal_x, goal_y = goal_pos
            if self.ogm and (goal_x, goal_y) not in self.ogm.goals:
                self.ogm.mark_goal(goal_x, goal_y)
                debug_log(f"LIDAR: NEW goal discovered at ({goal_x}, {goal_y})")
        
        # Update OGM with discovered loading dock
        if scan_results['loading_dock']:
            dock_x, dock_y = scan_results['loading_dock']
            if self.ogm and self.ogm.loading_dock != (dock_x, dock_y):
                self.ogm.mark_loading_dock(dock_x, dock_y)
                debug_log(f"LIDAR: NEW loading dock discovered at ({dock_x}, {dock_y})")
        
        # Update OGM with discovered discharge dock
        if scan_results['discharge_dock']:
            dock_x, dock_y = scan_results['discharge_dock']
            if self.ogm and self.ogm.discharge_dock != (dock_x, dock_y):
                self.ogm.mark_discharge_dock(dock_x, dock_y)
                debug_log(f"LIDAR: NEW discharge dock discovered at ({dock_x}, {dock_y})")
        
        # Update SLAM with sensor data
        self.slam.update_pose_from_sensor(scan_results, self.ogm)
        
        # Store pose and scan for loop closure detection
        self.slam.store_pose_and_scan(scan_results)
        
        # Check pose error and reset SLAM pose if drift is too high
        # Reset more aggressively - any drift > 2 cells should be corrected
        estimated_pose = self.slam.get_estimated_pose()
        pose_error = math.sqrt((self.x - estimated_pose[0])**2 + (self.y - estimated_pose[1])**2)
        if pose_error > 2.0:  # Lowered from 5.0 to 2.0 for more aggressive correction
            debug_log(f"SLAM pose error too high ({pose_error:.2f} cells), resetting to actual position")
            self.slam.set_pose(self.x, self.y, self.rotation_angle)
        
        # Check for loop closure periodically (every 10 scans) to avoid false positives
        # Loop closure is for SLAM pose correction only - does NOT stop exploration
        if len(self.slam.pose_history) - self.last_loop_closure_check >= 10:
            loop_detected, matched_index = self.slam.detect_loop_closure(scan_results)
            self.last_loop_closure_check = len(self.slam.pose_history)
            
            if loop_detected:
                self.loop_closure_detected = True
                estimated_pose = self.slam.get_estimated_pose()
                self.loop_closure_position = estimated_pose[:2]  # Store x, y
                self.slam.correct_pose_loop_closure(matched_index)
                debug_log(f"SLAM: Loop closure detected and corrected at ({estimated_pose[0]:.2f}, {estimated_pose[1]:.2f}) - exploration CONTINUES")
                # Note: Loop closure is just a pose correction - exploration continues regardless
            else:
                self.loop_closure_detected = False
    
    def can_move_to(self, new_x, new_y, warehouse):
        """
        Check if robot can move to a new position.
        Enforces physics - robot can NEVER move through obstacles.
        ALWAYS uses both warehouse obstacles (ground truth) and OGM obstacles.
        
        Args:
            new_x: Target x position
            new_y: Target y position
            warehouse: Warehouse object to check obstacles
        
        Returns:
            bool: True if movement is valid (no obstacle, in bounds)
        """
        # Check bounds first
        if new_x < 0 or new_x >= WAREHOUSE_WIDTH or new_y < 0 or new_y >= WAREHOUSE_HEIGHT:
            return False
        
        # CRITICAL: Check warehouse obstacle (ground truth - absolute authority)
        if warehouse.is_blocked(int(new_x), int(new_y)):
            return False
        
        # ALWAYS check OGM obstacles - OGM is part of the SLAM system and must be used
        if self.ogm and self.ogm.is_obstacle(int(new_x), int(new_y)):
            return False
        
        # ALWAYS check path is clear using LIDAR - if path is blocked, movement is blocked
        if not self.lidar.check_path_clear(self.x, self.y, new_x, new_y, warehouse):
            return False
        
        return True
    
    def move_to(self, new_x, new_y, warehouse, current_time):
        """
        Move robot to a new position with full physics validation.
        This is the ONLY way the robot should change position.
        
        Args:
            new_x: Target x position
            new_y: Target y position
            warehouse: Warehouse object
            current_time: Current game time
        
        Returns:
            bool: True if movement succeeded, False if blocked
        """
        # CRITICAL: Always validate movement before executing
        if not self.can_move_to(new_x, new_y, warehouse):
            debug_log(f"PHYSICS: Movement blocked to ({new_x}, {new_y}) - obstacle or out of bounds")
            return False
        
        # Calculate movement delta for odometry
        dx_actual = new_x - self.x
        dy_actual = new_y - self.y
        dtheta_actual = (self.target_rotation - self.rotation_angle) % 360
        if dtheta_actual > 180:
            dtheta_actual -= 360
        
        # Move robot (ground truth)
        old_x, old_y = self.x, self.y
        self.x = new_x
        self.y = new_y
        self.last_move_time = current_time
        
        # Verify we didn't end up on an obstacle (safety check)
        if warehouse.is_blocked(int(self.x), int(self.y)):
            # This should NEVER happen, but if it does, revert
            debug_log(f"PHYSICS ERROR: Robot ended up on obstacle at ({self.x}, {self.y})! Reverting.")
            self.x, self.y = old_x, old_y
            return False
        
        # Update odometry and SLAM
        odometry_delta = self.odometry.update_odometry(dx_actual, dy_actual, dtheta_actual)
        self.slam.predict_pose(odometry_delta)
        
        # Add to trajectory
        self.pose_trajectory.append((self.x, self.y))
        if len(self.pose_trajectory) > 1000:
            self.pose_trajectory.pop(0)
        
        # Sense surroundings after moving
        self.sense_surroundings(warehouse)
        
        return True
    
    def handle_input(self, keys, warehouse, current_time):
        """Handle keyboard input for robot movement with strict obstacle checking."""
        # Only allow movement if enough time has passed
        if current_time - self.last_move_time < self.move_cooldown:
            return
        
        new_x, new_y = self.x, self.y
        moved = False
        direction = None
        
        if keys[pygame.K_UP] and self.y > 0:
            new_y -= self.movement_speed
            moved = True
            direction = "UP"
            self.rotate_to(90)
        elif keys[pygame.K_DOWN] and self.y < WAREHOUSE_HEIGHT - 1:
            new_y += self.movement_speed
            moved = True
            direction = "DOWN"
            self.rotate_to(270)
        elif keys[pygame.K_LEFT] and self.x > 0:
            new_x -= self.movement_speed
            moved = True
            direction = "LEFT"
            self.rotate_to(180)
        elif keys[pygame.K_RIGHT] and self.x < WAREHOUSE_WIDTH - 1:
            new_x += self.movement_speed
            moved = True
            direction = "RIGHT"
            self.rotate_to(0)
        
        # Use centralized movement validation and execution
        if moved:
            if self.move_to(new_x, new_y, warehouse, current_time):
                debug_log(f"Robot moved {direction}: ({self.x}, {self.y})")
            else:
                debug_log(f"Movement blocked {direction} to ({new_x}, {new_y})")
    
    def start_mapping(self, warehouse):
        """
        Start autonomous mapping phase.
        Robot will explore the entire grid cell by cell using BFS.
        
        Args:
            warehouse: Warehouse object to map
        """
        self.warehouse = warehouse
        
        # Initialize OGM if not already done
        if self.ogm is None:
            self.ogm = OccupancyGridMap(WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT)
        
        debug_log("=" * 50)
        debug_log("STARTING AUTONOMOUS MAPPING PHASE")
        debug_log("Robot will explore the entire grid cell by cell...")
        debug_log("=" * 50)
        
        # Reset SLAM and odometry to starting position
        self.odometry.reset_odometry()
        self.slam.reset()
        self.odometry.set_pose(self.x, self.y, 0)
        self.slam.set_pose(self.x, self.y, 0)
        
        self.is_mapping = True
        self.mapping_complete = False
        self.visited = set()
        self.exploration_queue = []  # BFS queue for systematic exploration
        self.queued_cells = set()  # Track cells already in queue to avoid duplicates
        self.pose_trajectory = [(self.x, self.y)]
        self.loop_closure_detected = False
        self.last_loop_closure_check = 0  # Track when we last checked for loop closure
        
        # Faster movement during mapping for quicker exploration
        self.move_cooldown = 100  # milliseconds between moves (faster exploration)
        
        # Initialize OGM - mark all known obstacles from warehouse first
        for obs_x, obs_y in warehouse.obstacles:
            self.ogm.mark_obstacle(obs_x, obs_y)
            debug_log(f"Pre-marked obstacle at ({obs_x}, {obs_y}) from warehouse")
        
        # Mark starting position as visited and explored
        start_pos = (int(self.x), int(self.y))
        self.visited.add(start_pos)
        if not warehouse.is_blocked(int(self.x), int(self.y)):
            # Mark as explored in OGM (this also marks as free)
            self.ogm.mark_explored(int(self.x), int(self.y))
        
        # Sense surroundings at starting position
        self.sense_surroundings(warehouse)
        
        # Add adjacent unvisited cells to queue (only free cells, not obstacles)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        for dx, dy in directions:
            next_x = int(self.x) + dx
            next_y = int(self.y) + dy
            if 0 <= next_x < WAREHOUSE_WIDTH and 0 <= next_y < WAREHOUSE_HEIGHT:
                # CRITICAL: Check warehouse FIRST - never queue obstacle cells
                if warehouse.is_blocked(next_x, next_y):
                    continue  # Skip obstacle cells immediately
                
                # Also check OGM if it exists
                if self.ogm.is_obstacle(next_x, next_y):
                    continue  # Skip obstacle cells from OGM
                
                next_pos = (next_x, next_y)
                if next_pos not in self.visited and next_pos not in self.queued_cells:
                    self.exploration_queue.append(next_pos)
                    self.queued_cells.add(next_pos)
        
        debug_log(f"Starting SLAM exploration from position ({int(self.x)}, {int(self.y)})")
        total_free = len(warehouse.get_free_cells())
        debug_log(f"Total free cells to explore: {total_free}")
        debug_log(f"Initial queue size: {len(self.exploration_queue)}")
    
    def explore_next_cell(self, current_time):
        """
        Simple BFS exploration: visit every free cell, one at a time.
        
        Args:
            current_time: Current game time for movement cooldown
        
        Returns:
            bool: True if moved, False if mapping complete or waiting
        """
        if not self.is_mapping:
            return False
        
        if current_time - self.last_move_time < self.move_cooldown:
            return False
        
        # Get current position
        current_x = int(self.x)
        current_y = int(self.y)
        current_pos = (current_x, current_y)
        
        # Sense surroundings and mark current cell as visited and explored
        self.sense_surroundings(self.warehouse)
        if current_pos not in self.visited:
            self.visited.add(current_pos)
            if not self.warehouse.is_blocked(current_x, current_y):
                # Mark as explored in OGM (this also marks as free)
                if self.ogm:
                    self.ogm.mark_explored(current_x, current_y)
        
        # Add adjacent free unvisited cells to queue
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in directions:
            next_x = current_x + dx
            next_y = current_y + dy
            
            if 0 <= next_x < WAREHOUSE_WIDTH and 0 <= next_y < WAREHOUSE_HEIGHT:
                if (self.warehouse.is_blocked(next_x, next_y) or 
                    (self.ogm and self.ogm.is_obstacle(next_x, next_y))):
                    continue  # Skip obstacles
                
                next_pos = (next_x, next_y)
                if next_pos not in self.visited and next_pos not in self.queued_cells:
                    self.exploration_queue.append(next_pos)
                    self.queued_cells.add(next_pos)
        
        # If queue is empty, find frontiers (unvisited cells adjacent to explored cells)
        if not self.exploration_queue:
            total_free = self.warehouse.get_free_cells()
            unvisited = [cell for cell in total_free if cell not in self.visited]
            
            if not unvisited:
                # Check if all obstacles and goals are discovered before completing
                total_obstacles = len(self.warehouse.obstacles)
                obstacles_found = len(self.ogm.obstacles) if self.ogm else 0
                total_goals = len(self.warehouse.goals)
                goals_found = len(self.ogm.goals) if self.ogm else 0
                
                # Only complete if ALL free cells visited AND all obstacles discovered AND all goals discovered
                if obstacles_found >= total_obstacles and goals_found >= total_goals:
                    total_free_count = len(total_free)
                    debug_log(f"MAPPING COMPLETE! Visited {len(self.visited)}/{total_free_count} free cells, "
                             f"found {obstacles_found}/{total_obstacles} obstacles, "
                             f"{goals_found}/{total_goals} goals")
                    self.finish_mapping()
                    return False
                else:
                    # Still missing obstacles or goals - continue exploring
                    debug_log(f"All cells visited but missing: {total_obstacles - obstacles_found} obstacles, "
                             f"{total_goals - goals_found} goals - continuing exploration")
                    # Continue to try finding paths to unvisited cells (in case some are still reachable)
            
            # Find ALL frontiers (unvisited cells adjacent to explored cells)
            # For frontier detection, only check warehouse obstacles (ground truth)
            # OGM obstacles might have errors, so we'll find frontiers and validate with OGM during movement
            frontiers = []
            debug_log(f"Queue empty - searching for frontiers among {len(unvisited)} unvisited cells")
            
            for cell in unvisited:
                ux, uy = cell
                # Skip if marked as obstacle in warehouse (ground truth)
                if self.warehouse.is_blocked(ux, uy):
                    continue
                # Don't check OGM obstacles here - OGM might have errors, we'll validate during movement
                # This allows us to find frontiers even if OGM incorrectly marked them as obstacles
                
                # Check if adjacent to explored cell
                for dx, dy in directions:
                    adj_x, adj_y = ux + dx, uy + dy
                    if (adj_x, adj_y) in self.visited:
                        frontiers.append(cell)
                        break
            
            # If we found frontiers, add them to queue
            if frontiers:
                # Sort by distance from current position
                frontiers.sort(key=lambda c: abs(c[0] - current_x) + abs(c[1] - current_y))
                added_frontiers = 0
                for cell in frontiers[:20]:  # Increased from 10 to 20 to add more frontiers
                    if cell not in self.queued_cells:
                        self.exploration_queue.append(cell)
                        self.queued_cells.add(cell)
                        added_frontiers += 1
                debug_log(f"Queue empty - found {len(frontiers)} frontiers, added {added_frontiers} to queue")
            
            # If no frontiers found, try pathfinding to nearest unvisited cell using relaxed pathfinding
            if not self.exploration_queue:
                debug_log(f"No frontiers found, trying pathfinding to nearest unvisited cell")
                nearest = None
                min_dist = float('inf')
                for cell in unvisited:
                    ux, uy = cell
                    # Only check warehouse obstacles, not OGM (relaxed pathfinding will handle OGM)
                    if self.warehouse.is_blocked(ux, uy):
                        continue
                    
                    # Distance from current position
                    dist = abs(ux - current_x) + abs(uy - current_y)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = cell
                
                if nearest:
                    debug_log(f"Trying pathfinding to nearest unvisited cell {nearest} (distance: {min_dist})")
                    # Try relaxed pathfinding first (ignores OGM obstacles for pathfinding)
                    path = self._find_path_to_cell_relaxed(current_pos, nearest)
                    if not path:
                        # Fallback to normal pathfinding
                        path = self._find_path_to_cell(current_pos, nearest)
                    if path:
                        # Add path to queue (skip first cell - it's current position)
                        added_count = 0
                        for cell in path[1:]:
                            if cell not in self.queued_cells:
                                # Validate cell before adding (warehouse obstacles only, OGM checked in can_move_to)
                                # Allow visited cells in path - they might be needed to reach unvisited cells
                                if not self.warehouse.is_blocked(cell[0], cell[1]):
                                    self.exploration_queue.append(cell)
                                    self.queued_cells.add(cell)
                                    added_count += 1
                        debug_log(f"Queue empty - found path to nearest unvisited cell {nearest} (distance: {min_dist}), added {added_count} cells to queue")
                    else:
                        debug_log(f"Queue empty - no path found to nearest unvisited cell {nearest}")
                else:
                    debug_log(f"Queue empty - no valid unvisited cells found for pathfinding")
            
            # If still empty after all attempts, try to get unstuck by allowing revisiting nearby visited cells
            if not self.exploration_queue:
                # Try to find ANY reachable cell (even visited) to get unstuck
                debug_log(f"Queue empty after recovery attempts, trying to find any reachable cell to get unstuck")
                for dx, dy in directions:
                    check_x = current_x + dx
                    check_y = current_y + dy
                    
                    if (0 <= check_x < WAREHOUSE_WIDTH and 0 <= check_y < WAREHOUSE_HEIGHT):
                        if (not self.warehouse.is_blocked(check_x, check_y) and 
                            not (self.ogm and self.ogm.is_obstacle(check_x, check_y)) and
                            (check_x, check_y) not in self.queued_cells and
                            self.can_move_to(check_x, check_y, self.warehouse)):
                            
                            # Found a reachable cell (even if visited) - add to queue and rotate
                            self.exploration_queue.append((check_x, check_y))
                            self.queued_cells.add((check_x, check_y))
                            self.rotate_towards(check_x - current_x, check_y - current_y)
                            debug_log(f"Queue empty - found reachable cell ({check_x}, {check_y}) to get unstuck")
                            break
                
                # If still empty, return False to let main loop handle completion check
                if not self.exploration_queue:
                    return False  # Wait for next frame, main loop will check completion properly
        
        # Get next cell from queue (skip obstacles, but allow revisiting visited cells to get unstuck)
        attempts = 0
        max_attempts = len(self.exploration_queue) * 2  # Allow more attempts to find valid cells
        while self.exploration_queue and attempts < max_attempts:
            next_cell = self.exploration_queue.pop(0)
            self.queued_cells.discard(next_cell)
            attempts += 1
            
            nx, ny = next_cell
            if self.warehouse.is_blocked(nx, ny) or (self.ogm and self.ogm.is_obstacle(nx, ny)):
                continue  # Skip obstacles
            # Allow revisiting visited cells when queue is getting exhausted - helps get unstuck
            if next_cell in self.visited and len(self.exploration_queue) > 5:
                continue  # Skip visited cells if we have plenty of unvisited options
            
            # Move to this cell - validate with can_move_to first (includes LIDAR path check)
            if not self.can_move_to(nx, ny, self.warehouse):
                # Path blocked - rotate to explore adjacent cells and find new path
                # Allow revisiting visited cells when stuck to get unstuck
                directions_to_check = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, down, left, right
                found_alternative = False
                
                # First try: find unvisited adjacent cells
                for dx, dy in directions_to_check:
                    check_x = current_x + dx
                    check_y = current_y + dy
                    
                    if (0 <= check_x < WAREHOUSE_WIDTH and 0 <= check_y < WAREHOUSE_HEIGHT):
                        if (not self.warehouse.is_blocked(check_x, check_y) and 
                            not (self.ogm and self.ogm.is_obstacle(check_x, check_y)) and
                            (check_x, check_y) not in self.visited and
                            (check_x, check_y) not in self.queued_cells and
                            self.can_move_to(check_x, check_y, self.warehouse)):
                            
                            # Found an alternative adjacent cell - add to queue and rotate towards it
                            self.exploration_queue.append((check_x, check_y))
                            self.queued_cells.add((check_x, check_y))
                            self.rotate_towards(check_x - current_x, check_y - current_y)
                            found_alternative = True
                            debug_log(f"Path blocked to ({nx}, {ny}), found alternative adjacent cell ({check_x}, {check_y})")
                            break
                
                # Second try: if no unvisited cells found, allow revisiting visited cells to get unstuck
                if not found_alternative:
                    for dx, dy in directions_to_check:
                        check_x = current_x + dx
                        check_y = current_y + dy
                        
                        if (0 <= check_x < WAREHOUSE_WIDTH and 0 <= check_y < WAREHOUSE_HEIGHT):
                            if (not self.warehouse.is_blocked(check_x, check_y) and 
                                not (self.ogm and self.ogm.is_obstacle(check_x, check_y)) and
                                (check_x, check_y) not in self.queued_cells and
                                self.can_move_to(check_x, check_y, self.warehouse)):
                                
                                # Found an alternative (even if visited) - add to queue and rotate towards it
                                self.exploration_queue.append((check_x, check_y))
                                self.queued_cells.add((check_x, check_y))
                                self.rotate_towards(check_x - current_x, check_y - current_y)
                                found_alternative = True
                                debug_log(f"Path blocked to ({nx}, {ny}), found alternative visited cell ({check_x}, {check_y}) to get unstuck")
                                break
                
                if not found_alternative:
                    # No alternative found - mark as obstacle in OGM if it's not already
                    if not self.warehouse.is_blocked(nx, ny) and self.ogm and not self.ogm.is_obstacle(nx, ny):
                        # Mark as obstacle in OGM so we don't try again
                        self.ogm.mark_obstacle(nx, ny)
                        debug_log(f"Marked unreachable cell ({nx}, {ny}) as obstacle in OGM")
                    # Force rotation to scan surroundings for new paths
                    self.rotation_angle = (self.rotation_angle + 90) % 360
                    self.target_rotation = self.rotation_angle
                    debug_log(f"Path blocked to ({nx}, {ny}), rotating to scan surroundings")
                
                continue  # Skip this cell, try next in queue
            
            # Cell is reachable, try to move
            dx = nx - current_x
            dy = ny - current_y
            self.rotate_towards(dx, dy)
            
            if self.move_to(nx, ny, self.warehouse, current_time):
                # Log progress periodically
                if len(self.visited) % 50 == 0 or len(self.visited) <= 10:
                    total_free = len(self.warehouse.get_free_cells())
                    remaining = total_free - len(self.visited)
                    debug_log(f"Cell ({nx}, {ny}) | Visited: {len(self.visited)}/{total_free} | Remaining: {remaining}")
                return True
        
        # Queue exhausted or all cells failed - try to get unstuck first, then find new unvisited cells
        if not self.exploration_queue:
            # Try to find ANY reachable adjacent cell (even visited) to get unstuck
            debug_log(f"Queue exhausted - trying to find any reachable cell to get unstuck")
            for dx, dy in directions:
                check_x = current_x + dx
                check_y = current_y + dy
                
                if (0 <= check_x < WAREHOUSE_WIDTH and 0 <= check_y < WAREHOUSE_HEIGHT):
                    if (not self.warehouse.is_blocked(check_x, check_y) and 
                        not (self.ogm and self.ogm.is_obstacle(check_x, check_y)) and
                        (check_x, check_y) not in self.queued_cells and
                        self.can_move_to(check_x, check_y, self.warehouse)):
                        
                        # Found a reachable cell (even if visited) - add to queue and rotate
                        self.exploration_queue.append((check_x, check_y))
                        self.queued_cells.add((check_x, check_y))
                        self.rotate_towards(check_x - current_x, check_y - current_y)
                        debug_log(f"Queue exhausted - found reachable cell ({check_x}, {check_y}) to get unstuck")
                        # Try to process immediately
                        if self.exploration_queue:
                            next_cell = self.exploration_queue.pop(0)
                            self.queued_cells.discard(next_cell)
                            nx, ny = next_cell
                            if (not self.warehouse.is_blocked(nx, ny) and 
                                not (self.ogm and self.ogm.is_obstacle(nx, ny)) and
                                self.can_move_to(nx, ny, self.warehouse)):
                                dx = nx - current_x
                                dy = ny - current_y
                                self.rotate_towards(dx, dy)
                                if self.move_to(nx, ny, self.warehouse, current_time):
                                    return True
                        break
            
            # If still empty, try to find new unvisited cells immediately
            if not self.exploration_queue:
                total_free = self.warehouse.get_free_cells()
                unvisited = [cell for cell in total_free if cell not in self.visited]
            
            if not unvisited:
                # Check if all obstacles and goals are discovered
                total_obstacles = len(self.warehouse.obstacles)
                obstacles_found = len(self.ogm.obstacles) if self.ogm else 0
                total_goals = len(self.warehouse.goals)
                goals_found = len(self.ogm.goals) if self.ogm else 0
                
                # Only complete if ALL free cells visited AND all obstacles discovered AND all goals discovered
                if obstacles_found >= total_obstacles and goals_found >= total_goals:
                    # All cells visited and all obstacles/goals discovered - mapping complete
                    total_free_count = len(total_free)
                    debug_log(f"MAPPING COMPLETE! Visited {len(self.visited)}/{total_free_count} free cells, "
                             f"found {obstacles_found}/{total_obstacles} obstacles, "
                             f"{goals_found}/{total_goals} goals")
                    self.finish_mapping()
                    return False
                else:
                    # Still missing obstacles or goals - continue exploring
                    debug_log(f"All cells visited but missing: {total_obstacles - obstacles_found} obstacles, "
                             f"{total_goals - goals_found} goals - continuing exploration")
            
            # Try nearest unvisited cells until we find one we can reach
            # Sort by distance to try nearest first
            sorted_unvisited = sorted(unvisited, key=lambda c: abs(c[0] - current_x) + abs(c[1] - current_y))
            
            for nearest in sorted_unvisited[:20]:  # Try up to 20 nearest cells
                ux, uy = nearest
                
                # Skip if already marked as obstacle
                if self.warehouse.is_blocked(ux, uy) or self.ogm.is_obstacle(ux, uy):
                    continue
                
                # Check if we can actually reach this cell
                if not self.can_move_to(ux, uy, self.warehouse):
                    # Don't mark as obstacle immediately - might be temporary issue
                    # Only mark if we've tried multiple times
                    continue
                
                # Try to find path using relaxed pathfinding first
                path = self._find_path_to_cell_relaxed(current_pos, nearest)
                if not path:
                    # Fallback to normal pathfinding
                    path = self._find_path_to_cell(current_pos, nearest)
                if not path:
                    # Still no path, but don't mark as obstacle - might be reachable later
                    continue
                
                # Validate path - check each step is reachable
                valid_path = []
                prev_cell = current_pos
                for cell in path[1:]:  # Skip first cell (current position)
                    # Check if we can move from previous cell to this cell
                    if not self.can_move_to(cell[0], cell[1], self.warehouse):
                        # Cell unreachable, but don't mark as obstacle - might be temporary
                        break
                    # Check LIDAR path clearance
                    if not self.lidar.check_path_clear(prev_cell[0], prev_cell[1], cell[0], cell[1], self.warehouse):
                        # Path blocked, stop here
                        break
                    valid_path.append(cell)
                    prev_cell = cell
                
                if valid_path:
                    # Add valid path to queue
                    for cell in valid_path:
                        if cell not in self.queued_cells and cell not in self.visited:
                            self.exploration_queue.append(cell)
                            self.queued_cells.add(cell)
                    debug_log(f"Queue exhausted - found valid path to cell {nearest} ({len(valid_path)} steps)")
                    # Try to process one cell from the new queue immediately
                    if self.exploration_queue:
                        next_cell = self.exploration_queue.pop(0)
                        self.queued_cells.discard(next_cell)
                        nx, ny = next_cell
                        if (not self.warehouse.is_blocked(nx, ny) and 
                            not self.ogm.is_obstacle(nx, ny) and
                            next_cell not in self.visited and
                            self.can_move_to(nx, ny, self.warehouse)):
                            dx = nx - current_x
                            dy = ny - current_y
                            self.rotate_towards(dx, dy)
                            if self.move_to(nx, ny, self.warehouse, current_time):
                                return True
                    break  # Found valid path, exit loop
                else:
                    # Path invalid - mark as obstacle and try next
                    if not self.warehouse.is_blocked(ux, uy) and self.ogm:
                        self.ogm.mark_obstacle(ux, uy)
                    continue  # Try next nearest cell
            
            # If we got here, couldn't find a valid path - try adjacent unvisited cells as fallback
            for cell in unvisited[:10]:  # Limit to first 10 to avoid infinite loop
                ux, uy = cell
                if self.warehouse.is_blocked(ux, uy) or self.ogm.is_obstacle(ux, uy):
                    continue
                for dx, dy in directions:
                    adj_x, adj_y = ux + dx, uy + dy
                    if (adj_x, adj_y) in self.visited:
                        if cell not in self.queued_cells and self.can_move_to(cell[0], cell[1], self.warehouse):
                            self.exploration_queue.append(cell)
                            self.queued_cells.add(cell)
                        break
            # Try to process one cell from the new queue immediately
            if self.exploration_queue:
                next_cell = self.exploration_queue.pop(0)
                self.queued_cells.discard(next_cell)
                nx, ny = next_cell
                if (not self.warehouse.is_blocked(nx, ny) and 
                    not self.ogm.is_obstacle(nx, ny) and
                    next_cell not in self.visited and
                    self.can_move_to(nx, ny, self.warehouse)):
                    dx = nx - current_x
                    dy = ny - current_y
                    self.rotate_towards(dx, dy)
                    if self.move_to(nx, ny, self.warehouse, current_time):
                        return True
        
        # No valid cell found, wait for next frame
        return False
    
    def _find_path_to_cell(self, start, target):
        """
        Simple BFS pathfinding to find path from start to target.
        Returns list of cells forming the path, or None if unreachable.
        
        Args:
            start: (x, y) starting position
            target: (x, y) target position
        
        Returns:
            list: Path as list of (x, y) tuples, or None if unreachable
        """
        if start == target:
            return [start]
        
        sx, sy = start
        tx, ty = target
        
        # Check if target is valid
        if (tx < 0 or tx >= WAREHOUSE_WIDTH or ty < 0 or ty >= WAREHOUSE_HEIGHT or
            self.warehouse.is_blocked(tx, ty) or self.ogm.is_obstacle(tx, ty)):
            return None
        
        # BFS pathfinding with limit to prevent infinite loops
        max_pathfinding_steps = 1000  # Limit pathfinding to reasonable size
        queue = [(sx, sy)]
        visited = {start}
        parent = {start: None}
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        steps = 0
        
        while queue and steps < max_pathfinding_steps:
            current = queue.pop(0)
            cx, cy = current
            steps += 1
            
            if current == target:
                # Reconstruct path
                path = []
                node = target
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            
            # Check neighbors
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                next_cell = (nx, ny)
                
                if (nx < 0 or nx >= WAREHOUSE_WIDTH or ny < 0 or ny >= WAREHOUSE_HEIGHT):
                    continue
                
                if (self.warehouse.is_blocked(nx, ny) or self.ogm.is_obstacle(nx, ny)):
                    continue
                
                if next_cell in visited:
                    continue
                
                visited.add(next_cell)
                parent[next_cell] = current
                queue.append(next_cell)
        
        # No path found (either unreachable or path too long)
        return None
    
    def _find_path_to_cell_relaxed(self, start, target):
        """
        Relaxed BFS pathfinding that only checks warehouse obstacles, not OGM.
        NOTE: This is ONLY for pathfinding planning. Actual movement validation
        ALWAYS uses OGM through can_move_to(). This helps find potential paths
        when OGM might have errors, but movement is still validated against OGM.
        
        Args:
            start: (x, y) starting position
            target: (x, y) target position
        
        Returns:
            list: Path as list of (x, y) tuples, or None if unreachable
        """
        if start == target:
            return [start]
        
        sx, sy = start
        tx, ty = target
        
        # Only check warehouse obstacles, not OGM (which might have errors)
        if (tx < 0 or tx >= WAREHOUSE_WIDTH or ty < 0 or ty >= WAREHOUSE_HEIGHT or
            self.warehouse.is_blocked(tx, ty)):
            return None
        
        # BFS pathfinding
        max_pathfinding_steps = 1000
        queue = [(sx, sy)]
        visited = {start}
        parent = {start: None}
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        steps = 0
        
        while queue and steps < max_pathfinding_steps:
            current = queue.pop(0)
            cx, cy = current
            steps += 1
            
            if current == target:
                # Reconstruct path
                path = []
                node = target
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            
            # Check neighbors - only check warehouse obstacles, not OGM
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                next_cell = (nx, ny)
                
                if (nx < 0 or nx >= WAREHOUSE_WIDTH or ny < 0 or ny >= WAREHOUSE_HEIGHT):
                    continue
                
                # RELAXED: Only check warehouse obstacles, not OGM
                if self.warehouse.is_blocked(nx, ny):
                    continue
                
                if next_cell in visited:
                    continue
                
                queue.append(next_cell)
                visited.add(next_cell)
                parent[next_cell] = current
        
        return None
    
    def finish_mapping(self):
        """Finish the mapping phase and print results."""
        self.is_mapping = False
        self.mapping_complete = True
        
        # Sense surroundings one final time
        self.sense_surroundings(self.warehouse)
        
        # Verify complete mapping - all free cells visited, all obstacles discovered, all goals discovered
        total_free = self.warehouse.get_free_cells()
        unvisited = [cell for cell in total_free if cell not in self.visited]
        total_obstacles = len(self.warehouse.obstacles)
        obstacles_found = len(self.ogm.obstacles)
        total_goals = len(self.warehouse.goals)
        goals_found = len(self.ogm.goals)
        
        # Print stored locations
        debug_log("")
        debug_log("=" * 50)
        debug_log("MAPPING COMPLETE - Stored Locations:")
        debug_log("=" * 50)
        debug_log(f"Total free cells: {len(total_free)}")
        debug_log(f"Cells visited: {len(self.visited)}")
        if unvisited:
            debug_log(f"WARNING: {len(unvisited)} unvisited cells: {unvisited}")
        else:
            debug_log("✓ All free cells visited successfully!")
        debug_log(f"Total obstacles: {total_obstacles}")
        debug_log(f"Obstacles found: {obstacles_found}")
        if obstacles_found < total_obstacles:
            debug_log(f"WARNING: {total_obstacles - obstacles_found} obstacles not discovered!")
        else:
            debug_log("✓ All obstacles discovered!")
        debug_log(f"Total goals: {total_goals}")
        debug_log(f"Goals found: {goals_found}")
        if goals_found < total_goals:
            debug_log(f"WARNING: {total_goals - goals_found} goals not discovered!")
        else:
            debug_log("✓ All goals discovered!")
        debug_log(f"Goal locations: {sorted(self.ogm.goals)}")
        debug_log(f"Loading dock: {self.ogm.loading_dock}")
        debug_log(f"Discharge dock: {self.ogm.discharge_dock}")
        debug_log("=" * 50)
        debug_log("")
        
        # Return robot to starting position
        self.x = self.start_x
        self.y = self.start_y
        self.rotation_angle = 0
        self.target_rotation = 0
        
        # Reset SLAM and odometry
        self.odometry.reset_odometry()
        self.slam.reset()
        self.odometry.set_pose(self.x, self.y, 0)
        self.slam.set_pose(self.x, self.y, 0)
        
        # Restore normal move cooldown after mapping
        self.move_cooldown = 200  # milliseconds between moves (normal speed)
        
        debug_log(f"Robot returned to starting position ({self.start_x}, {self.start_y})")
        debug_log("Mapping complete - robot ready for manual control")
    
    def try_pickup(self, warehouse, current_time):
        """Try to pick up cargo at current goal location."""
        if current_time - self.last_action_time < self.action_cooldown:
            debug_log("Pickup cooldown active, waiting...")
            return False
        
        if self.has_cargo or not self.current_goal:
            debug_log(f"Cannot pickup - has_cargo: {self.has_cargo}, current_goal: {self.current_goal}")
            return False
        
        goal_x, goal_y = self.current_goal
        if abs(self.x - goal_x) < 1.5 and abs(self.y - goal_y) < 1.5:
            self.has_cargo = True
            warehouse.goals.remove(self.current_goal)
            self.last_action_time = current_time
            debug_log(f"Picked up cargo at goal ({goal_x}, {goal_y})")
            return True
        
        debug_log(f"Not close enough to goal - robot at ({self.x}, {self.y}), goal at ({goal_x}, {goal_y})")
        return False
    
    def try_drop(self, warehouse, current_time):
        """Try to drop cargo at discharge dock."""
        if current_time - self.last_action_time < self.action_cooldown:
            debug_log("Drop cooldown active, waiting...")
            return False
        
        if not self.has_cargo:
            debug_log("Cannot drop - no cargo to drop")
            return False
        
        dock_x, dock_y = warehouse.discharge_dock
        if abs(self.x - dock_x) < 1.5 and abs(self.y - dock_y) < 1.5:
            self.has_cargo = False
            self.score += 1  # Increment score for each delivery
            self.last_action_time = current_time
            debug_log(f"Dropped cargo at discharge dock ({dock_x}, {dock_y}). Score: {self.score}")
            # Update current goal to the next one
            if warehouse.goals:
                self.current_goal = warehouse.goals[0]
                debug_log(f"Next goal set to: {self.current_goal}")
            else:
                self.current_goal = None
                debug_log("No more goals remaining")
            return True
        
        debug_log(f"Not at discharge dock - robot at ({self.x}, {self.y}), dock at ({dock_x}, {dock_y})")
        return False
    
    def draw(self, surface):
        """Draw the robot with rotation, estimated pose, and SLAM visualization."""
        # Draw pose trajectory (path taken)
        if len(self.pose_trajectory) > 1:
            trajectory_points = [(int(x * GRID_SIZE + GRID_SIZE // 2), 
                                 int(y * GRID_SIZE + GRID_SIZE // 2)) 
                                for x, y in self.pose_trajectory[-100:]]  # Last 100 points
            if len(trajectory_points) > 1:
                pygame.draw.lines(surface, (200, 200, 200), False, trajectory_points, 2)
        
        # Draw estimated pose from SLAM (green circle)
        estimated_pose = self.slam.get_estimated_pose()
        est_x_pixel = estimated_pose[0] * GRID_SIZE
        est_y_pixel = estimated_pose[1] * GRID_SIZE
        est_center_x = est_x_pixel + GRID_SIZE // 2
        est_center_y = est_y_pixel + GRID_SIZE // 2
        
        # Draw uncertainty ellipse around estimated pose
        uncertainty = self.slam.get_uncertainty()
        pos_uncertainty = uncertainty[0]
        ellipse_radius = int(pos_uncertainty * GRID_SIZE * 2)
        if ellipse_radius > 0:
            pygame.draw.ellipse(
                surface,
                (150, 255, 150),
                (est_center_x - ellipse_radius, est_center_y - ellipse_radius,
                 ellipse_radius * 2, ellipse_radius * 2),
                1
            )
        
        # Draw estimated pose (smaller, green circle)
        pygame.draw.circle(
            surface,
            (0, 255, 0),  # Green for estimated
            (est_center_x, est_center_y),
            GRID_SIZE // 4
        )
        
        # Draw line from estimated to actual pose
        x_pixel = self.x * GRID_SIZE
        y_pixel = self.y * GRID_SIZE
        center_x = x_pixel + GRID_SIZE // 2
        center_y = y_pixel + GRID_SIZE // 2
        
        pygame.draw.line(
            surface,
            (255, 200, 0),  # Orange line connecting estimated to actual
            (est_center_x, est_center_y),
            (center_x, center_y),
            2
        )
        
        # Draw robot body (actual position - blue circle)
        pygame.draw.circle(
            surface,
            BLUE,
            (center_x, center_y),
            GRID_SIZE // 3
        )
        
        # Draw rotation indicator (line showing direction)
        angle_rad = math.radians(self.rotation_angle)
        line_length = GRID_SIZE // 2
        end_x = center_x + math.cos(angle_rad) * line_length
        end_y = center_y - math.sin(angle_rad) * line_length  # Negative because y increases downward
        pygame.draw.line(surface, BLACK, (center_x, center_y), (int(end_x), int(end_y)), 3)
        
        # Draw cargo indicator
        if self.has_cargo:
            pygame.draw.circle(
                surface,
                RED,
                (center_x, center_y),
                GRID_SIZE // 5
            )
        
        # Draw loop closure indicator if detected
        if self.loop_closure_detected and self.loop_closure_position:
            loop_x = int(self.loop_closure_position[0] * GRID_SIZE + GRID_SIZE // 2)
            loop_y = int(self.loop_closure_position[1] * GRID_SIZE + GRID_SIZE // 2)
            pygame.draw.circle(
                surface,
                (255, 0, 255),  # Magenta for loop closure
                (loop_x, loop_y),
                GRID_SIZE // 2,
                3
            )
