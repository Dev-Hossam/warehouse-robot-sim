"""
Warehouse module for the warehouse robot simulation.
"""

import math
import pygame
from constants import (
    WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT, GRID_SIZE, 
    GRAY, LOADING_DOCK, DISCHARGE_DOCK, GOAL_COLOR, 
    YELLOW, DARK_GRAY, BLACK, WHITE, debug_log
)

DEBUG = True

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[WAREHOUSE DEBUG] {message}")


class Warehouse:
    """Warehouse environment with obstacles, goals, and docks."""
    
    def __init__(self, map_name='map1', num_dynamic_obstacles=0, obstacle_speed=350):
        self.obstacles = set()
        self.goals = []
        self.loading_dock = None
        self.discharge_dock = None
        self.width = WAREHOUSE_WIDTH
        self.height = WAREHOUSE_HEIGHT
        self.map_name = map_name
        self.num_dynamic_obstacles = num_dynamic_obstacles  # Number of dynamic obstacles to spawn
        self.obstacle_speed = obstacle_speed  # Movement cooldown in milliseconds for dynamic obstacles
        self.dynamic_obstacles = []  # List of dynamic obstacles
        self.dynamic_obstacles_spawned = False  # Flag to track spawning
        # Performance optimization: cache background surface
        self.background_surface = None  # Cached background (grid + explored/unexplored cells)
        self.last_explored_cells = set()  # Track exploration changes
        # Red box (high-priority cargo) system
        self.red_box = None  # (x, y) position or None
        self.red_box_spawn_time = None  # Timestamp when red box was spawned
        self.red_box_expiration_time = 9000  # 9 seconds in milliseconds (increased from 7s)
        # Static red box locations for map7
        self.static_red_box_locations = []  # List of static red box positions for map7
        self.red_box_spawn_index = 0  # Current index in static_red_box_locations for map7
        self.total_red_boxes_spawned = 0  # Total number of red boxes spawned (for map7 limit)
        debug_log(f"Creating warehouse layout: {map_name}...")
        self.create_maze_layout(map_name)
        self.create_docks_and_goals(map_name)
        debug_log(f"Warehouse created with {len(self.obstacles)} obstacles and {len(self.goals)} goals")
        if num_dynamic_obstacles > 0:
            debug_log(f"Dynamic obstacles will be spawned: {num_dynamic_obstacles}")
    
    def create_maze_layout(self, map_name='map1'):
        """Create the maze layout with obstacles based on map name."""
        # Create walls around the perimeter
        for x in range(WAREHOUSE_WIDTH):
            self.obstacles.add((x, 0))
            self.obstacles.add((x, WAREHOUSE_HEIGHT - 1))
        for y in range(WAREHOUSE_HEIGHT):
            self.obstacles.add((0, y))
            self.obstacles.add((WAREHOUSE_WIDTH - 1, y))
        
        if map_name == 'map1':
            # Original map layout
            # Create internal maze structure with wider paths
            # Horizontal corridors (skip every 5th row for wider paths)
            for y in [4, 8, 12]:
                if y < WAREHOUSE_HEIGHT:
                    for x in range(2, WAREHOUSE_WIDTH - 2):
                        if x % 8 not in [0, 1]:  # Wider gaps for vertical passages
                            self.obstacles.add((x, y))
            
            # Vertical corridors (wider spacing)
            for x in [7, 15]:
                if x < WAREHOUSE_WIDTH:
                    for y in range(2, WAREHOUSE_HEIGHT - 2):
                        if y % 6 not in [0, 1]:  # Wider gaps for horizontal passages
                            self.obstacles.add((x, y))
            
            # Add strategic obstacles
            maze_obstacles = [
                (4, 6), (10, 6), (18, 6),
                (4, 10), (12, 10), (20, 10),
                (6, 14), (14, 14), (22, 14),
            ]
            for x, y in maze_obstacles:
                if x < WAREHOUSE_WIDTH and y < WAREHOUSE_HEIGHT:
                    self.obstacles.add((x, y))
        
        elif map_name == 'map2':
            # Different map layout - more open with grid-like obstacles
            # Create vertical walls with gaps
            for x in [5, 10, 15, 20]:
                if x < WAREHOUSE_WIDTH:
                    for y in range(2, WAREHOUSE_HEIGHT - 2):
                        if y % 4 not in [0, 1]:  # Gaps every 4 cells
                            self.obstacles.add((x, y))
            
            # Create horizontal walls with gaps
            for y in [3, 7, 11, 15]:
                if y < WAREHOUSE_HEIGHT:
                    for x in range(2, WAREHOUSE_WIDTH - 2):
                        if x % 4 not in [0, 1]:  # Gaps every 4 cells
                            self.obstacles.add((x, y))
            
            # Add some scattered obstacles
            scattered_obstacles = [
                (3, 5), (8, 5), (13, 5), (18, 5),
                (6, 9), (11, 9), (16, 9), (21, 9),
                (4, 13), (9, 13), (14, 13), (19, 13),
            ]
            for x, y in scattered_obstacles:
                if x < WAREHOUSE_WIDTH and y < WAREHOUSE_HEIGHT:
                    self.obstacles.add((x, y))
        
        elif map_name == 'map3':
            # Another different layout - more maze-like with winding paths
            # Create a complex maze pattern
            for y in [3, 6, 9, 12, 15]:
                if y < WAREHOUSE_HEIGHT:
                    for x in range(2, WAREHOUSE_WIDTH - 2):
                        if (x + y) % 5 not in [0, 1]:  # Pattern with gaps
                            self.obstacles.add((x, y))
            
            # Vertical walls with pattern
            for x in [6, 12, 18]:
                if x < WAREHOUSE_WIDTH:
                    for y in range(2, WAREHOUSE_HEIGHT - 2):
                        if (x + y) % 6 not in [0, 1, 2]:  # Different pattern
                            self.obstacles.add((x, y))
            
            # Add corner obstacles
            corner_obstacles = [
                (3, 4), (9, 4), (15, 4), (21, 4),
                (3, 8), (9, 8), (15, 8), (21, 8),
                (3, 12), (9, 12), (15, 12), (21, 12),
                (5, 6), (11, 6), (17, 6), (23, 6),
                (5, 10), (11, 10), (17, 10), (23, 10),
            ]
            for x, y in corner_obstacles:
                if x < WAREHOUSE_WIDTH and y < WAREHOUSE_HEIGHT:
                    self.obstacles.add((x, y))
        
        elif map_name == 'map4':
            # Map4 layout - similar to map1 but with different structure
            # Create internal maze structure with wider paths
            # Horizontal corridors
            for y in [4, 8, 12]:
                if y < WAREHOUSE_HEIGHT:
                    for x in range(2, WAREHOUSE_WIDTH - 2):
                        if x % 8 not in [0, 1]:  # Wider gaps for vertical passages
                            self.obstacles.add((x, y))
            
            # Vertical corridors (wider spacing)
            for x in [7, 15]:
                if x < WAREHOUSE_WIDTH:
                    for y in range(2, WAREHOUSE_HEIGHT - 2):
                        if y % 6 not in [0, 1]:  # Wider gaps for horizontal passages
                            self.obstacles.add((x, y))
            
            # Add strategic obstacles
            maze_obstacles = [
                (4, 6), (10, 6), (18, 6),
                (4, 10), (12, 10), (20, 10),
                (6, 14), (14, 14), (22, 14),
            ]
            for x, y in maze_obstacles:
                if x < WAREHOUSE_WIDTH and y < WAREHOUSE_HEIGHT:
                    self.obstacles.add((x, y))
        
        elif map_name == 'map6':
            # Map6 layout - replica of map4 with static goals
            # Create internal maze structure with wider paths
            # Horizontal corridors
            for y in [4, 8, 12]:
                if y < WAREHOUSE_HEIGHT:
                    for x in range(2, WAREHOUSE_WIDTH - 2):
                        if x % 8 not in [0, 1]:  # Wider gaps for vertical passages
                            self.obstacles.add((x, y))
            
            # Vertical corridors (wider spacing)
            for x in [7, 15]:
                if x < WAREHOUSE_WIDTH:
                    for y in range(2, WAREHOUSE_HEIGHT - 2):
                        if y % 6 not in [0, 1]:  # Wider gaps for horizontal passages
                            self.obstacles.add((x, y))
            
            # Add strategic obstacles
            maze_obstacles = [
                (4, 6), (10, 6), (18, 6),
                (4, 10), (12, 10), (20, 10),
                (6, 14), (14, 14), (22, 14),
            ]
            for x, y in maze_obstacles:
                if x < WAREHOUSE_WIDTH and y < WAREHOUSE_HEIGHT:
                    self.obstacles.add((x, y))
        
        elif map_name == 'map5':
            # Map5 layout - Red box feature map with more open design
            # Similar to map2 but with more space for dynamic cargo
            # Create vertical walls with gaps
            for x in [5, 10, 15, 20]:
                if x < WAREHOUSE_WIDTH:
                    for y in range(2, WAREHOUSE_HEIGHT - 2):
                        if y % 4 not in [0, 1]:  # Gaps every 4 cells
                            self.obstacles.add((x, y))
            
            # Create horizontal walls with gaps
            for y in [3, 7, 11, 15]:
                if y < WAREHOUSE_HEIGHT:
                    for x in range(2, WAREHOUSE_WIDTH - 2):
                        if x % 4 not in [0, 1]:  # Gaps every 4 cells
                            self.obstacles.add((x, y))
            
            # Add some scattered obstacles
            scattered_obstacles = [
                (3, 5), (8, 5), (13, 5), (18, 5),
                (6, 9), (11, 9), (16, 9), (21, 9),
                (4, 13), (9, 13), (14, 13), (19, 13),
            ]
            for x, y in scattered_obstacles:
                if x < WAREHOUSE_WIDTH and y < WAREHOUSE_HEIGHT:
                    self.obstacles.add((x, y))
        
        elif map_name == 'map7':
            # Map7 layout - Same as map5 but with static goals and static red box locations
            # Create vertical walls with gaps
            for x in [5, 10, 15, 20]:
                if x < WAREHOUSE_WIDTH:
                    for y in range(2, WAREHOUSE_HEIGHT - 2):
                        if y % 4 not in [0, 1]:  # Gaps every 4 cells
                            self.obstacles.add((x, y))
            
            # Create horizontal walls with gaps
            for y in [3, 7, 11, 15]:
                if y < WAREHOUSE_HEIGHT:
                    for x in range(2, WAREHOUSE_WIDTH - 2):
                        if x % 4 not in [0, 1]:  # Gaps every 4 cells
                            self.obstacles.add((x, y))
            
            # Add some scattered obstacles
            scattered_obstacles = [
                (3, 5), (8, 5), (13, 5), (18, 5),
                (6, 9), (11, 9), (16, 9), (21, 9),
                (4, 13), (9, 13), (14, 13), (19, 13),
            ]
            for x, y in scattered_obstacles:
                if x < WAREHOUSE_WIDTH and y < WAREHOUSE_HEIGHT:
                    self.obstacles.add((x, y))
    
    def create_docks_and_goals(self, map_name='map1'):
        """Create loading dock, discharge dock, and goal locations."""
        # Loading dock at top left area
        self.loading_dock = (2, 2)
        
        # Discharge dock at robot starting position
        self.discharge_dock = (1, 1)
        
        # For map6, use static goals (same as map4 layout but fixed positions)
        if map_name == 'map6':
            # Static goal positions for map6 (replica of map4 with fixed goals)
            # These positions are chosen to be in open areas and reachable
            static_goals = [
                (11, 2),   # Goal 1
                (19, 9),   # Goal 2
                (20, 9),   # Goal 3
                (2, 5),    # Goal 4
                (4, 7),    # Goal 5
                (21, 15),  # Goal 6
                (23, 14),  # Goal 7
                (8, 6),    # Goal 8
            ]
            # Validate static goals (filter out invalid positions)
            self.goals = []
            start_pos = self.discharge_dock if self.discharge_dock else (1, 1)
            for goal in static_goals:
                # Check if goal is valid (not on obstacle, not on dock, reachable)
                if (not self.is_blocked(goal[0], goal[1]) and
                    goal != self.discharge_dock and goal != self.loading_dock and
                    self.is_reachable(start_pos, goal)):
                    self.goals.append(goal)
                else:
                    debug_log(f"Warning: Static goal {goal} is invalid (on obstacle/dock or unreachable), skipping")
            debug_log(f"Using {len(self.goals)} static goals for map6: {self.goals}")
        elif map_name == 'map7':
            # Static goal positions for map7 (same layout as map5 but fixed positions)
            static_goals = [
                (7, 4),    # Goal 1
                (12, 4),   # Goal 2
                (17, 4),   # Goal 3
                (22, 4),   # Goal 4
                (4, 8),    # Goal 5
                (9, 8),    # Goal 6
                (14, 8),   # Goal 7
                (19, 8),   # Goal 8
            ]
            # Validate static goals (filter out invalid positions)
            self.goals = []
            start_pos = self.discharge_dock if self.discharge_dock else (1, 1)
            for goal in static_goals:
                # Check if goal is valid (not on obstacle, not on dock, reachable)
                if (not self.is_blocked(goal[0], goal[1]) and
                    goal != self.discharge_dock and goal != self.loading_dock and
                    self.is_reachable(start_pos, goal)):
                    self.goals.append(goal)
                else:
                    debug_log(f"Warning: Static goal {goal} is invalid (on obstacle/dock or unreachable), skipping")
            debug_log(f"Using {len(self.goals)} static goals for map7: {self.goals}")
            
            # Initialize static red box locations for map7 (10 locations)
            # These positions are chosen to be in open areas, reachable, and not overlapping with goals
            static_red_boxes = [
                (6, 6),    # Red box location 1
                (13, 6),   # Red box location 2
                (20, 6),   # Red box location 3
                (5, 10),   # Red box location 4
                (12, 10),  # Red box location 5
                (19, 10),  # Red box location 6
                (7, 12),   # Red box location 7
                (14, 12),  # Red box location 8
                (21, 12),  # Red box location 9
                (10, 14),  # Red box location 10
            ]
            # Validate static red box locations
            self.static_red_box_locations = []
            excluded_positions = {self.discharge_dock, self.loading_dock}
            if self.goals:
                excluded_positions.update(self.goals)
            for rb_loc in static_red_boxes:
                # Check if red box location is valid
                if (not self.is_blocked(rb_loc[0], rb_loc[1]) and
                    rb_loc not in excluded_positions and
                    self.is_reachable(start_pos, rb_loc)):
                    self.static_red_box_locations.append(rb_loc)
                else:
                    debug_log(f"Warning: Static red box location {rb_loc} is invalid, skipping")
            debug_log(f"Initialized {len(self.static_red_box_locations)} static red box locations for map7: {self.static_red_box_locations}")
        else:
            # Generate random goals with constraints for other maps
            self.goals = self.generate_random_goals(num_goals=8)
    
    def is_blocked(self, x, y):
        """Check if a position is blocked by an obstacle."""
        return (int(x), int(y)) in self.obstacles
    
    def get_all_cells(self):
        """Get all cells in the warehouse (for complete exploration)."""
        cells = []
        for y in range(self.height):
            for x in range(self.width):
                cells.append((x, y))
        return cells
    
    def get_free_cells(self):
        """Get all free (non-obstacle) cells in the warehouse."""
        free_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.is_blocked(x, y):
                    free_cells.append((x, y))
        return free_cells
    
    def is_reachable(self, start_pos, target_pos):
        """
        Check if target_pos is reachable from start_pos using BFS.
        
        Args:
            start_pos: (x, y) starting position
            target_pos: (x, y) target position
            
        Returns:
            bool: True if target is reachable from start
        """
        if start_pos == target_pos:
            return True
        
        if target_pos in self.obstacles:
            return False
        
        # BFS to check reachability
        from collections import deque
        
        queue = deque([start_pos])
        visited = {start_pos}
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
        
        while queue:
            current = queue.popleft()
            
            if current == target_pos:
                return True
            
            x, y = current
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                
                # Check bounds
                if not (0 <= nx < WAREHOUSE_WIDTH and 0 <= ny < WAREHOUSE_HEIGHT):
                    continue
                
                # Check if obstacle
                if neighbor in self.obstacles:
                    continue
                
                # Check if already visited
                if neighbor in visited:
                    continue
                
                visited.add(neighbor)
                queue.append(neighbor)
        
        return False
    
    def generate_random_goals(self, num_goals=8):
        """
        Generate random goal locations with constraints:
        1. Not on obstacles
        2. Within map bounds
        3. Reachable from discharge dock
        
        Args:
            num_goals: Number of goals to generate
            
        Returns:
            list: List of (x, y) goal positions
        """
        import random
        
        goals = []
        free_cells = self.get_free_cells()
        
        # Remove discharge dock and loading dock from free cells
        free_cells = [cell for cell in free_cells 
                     if cell != self.discharge_dock and cell != self.loading_dock]
        
        if len(free_cells) < num_goals:
            debug_log(f"Warning: Not enough free cells ({len(free_cells)}) for {num_goals} goals")
            num_goals = len(free_cells)
        
        start_pos = self.discharge_dock if self.discharge_dock else (1, 1)
        max_attempts = num_goals * 100  # Limit attempts
        attempts = 0
        
        while len(goals) < num_goals and attempts < max_attempts:
            attempts += 1
            
            # Randomly select a free cell
            candidate = random.choice(free_cells)
            
            # Check constraints
            # 1. Not on obstacle (already guaranteed by free_cells)
            # 2. Within bounds (already guaranteed by free_cells)
            # 3. Not already in goals
            if candidate in goals:
                continue
            
            # 4. Reachable from start position
            if not self.is_reachable(start_pos, candidate):
                continue
            
            # All constraints satisfied - add goal
            goals.append(candidate)
            debug_log(f"Generated goal {len(goals)} at {candidate}")
        
        if len(goals) < num_goals:
            debug_log(f"Warning: Only generated {len(goals)}/{num_goals} goals after {attempts} attempts")
        
        debug_log(f"Generated {len(goals)} random goals: {goals}")
        return goals
    
    def spawn_dynamic_obstacles(self, num_obstacles=8):
        """
        Spawn dynamic obstacles (workers and forklifts) at random free positions.
        Called when robot transitions to DELIVER_GOALS mode.
        
        Args:
            num_obstacles: Total number of obstacles to spawn
        """
        if self.dynamic_obstacles_spawned:
            return  # Already spawned
        
        import random
        from dynamic_obstacles import Worker, Forklift
        
        free_cells = self.get_free_cells()
        
        # Remove start position, loading dock, discharge dock, and goals from spawn locations
        excluded_positions = {self.discharge_dock, self.loading_dock}
        if self.goals:
            excluded_positions.update(self.goals)
        
        # Also exclude cells within 2 cells of discharge dock (Manhattan distance)
        if self.discharge_dock:
            dock_x, dock_y = self.discharge_dock
            for cell in free_cells[:]:  # Use slice to create a copy for iteration
                cell_x, cell_y = cell
                distance = abs(cell_x - dock_x) + abs(cell_y - dock_y)
                if distance <= 2:
                    excluded_positions.add(cell)
        
        free_cells = [cell for cell in free_cells if cell not in excluded_positions]
        
        if len(free_cells) < num_obstacles:
            debug_log(f"Warning: Not enough free cells ({len(free_cells)}) for {num_obstacles} obstacles")
            num_obstacles = len(free_cells)
        
        # Randomly assign types (worker or forklift)
        obstacle_types = []
        for i in range(num_obstacles):
            obstacle_types.append(random.choice(['worker', 'forklift']))
        
        # Spawn obstacles at random positions
        spawned_positions = set()
        for i, obs_type in enumerate(obstacle_types):
            max_attempts = 100
            attempts = 0
            while attempts < max_attempts:
                candidate = random.choice(free_cells)
                if candidate not in spawned_positions:
                    # Create obstacle with specified speed
                    if obs_type == 'worker':
                        obstacle = Worker(candidate[0], candidate[1], warehouse=self, move_cooldown=self.obstacle_speed)
                    else:  # forklift
                        obstacle = Forklift(candidate[0], candidate[1], warehouse=self, move_cooldown=self.obstacle_speed)
                    
                    self.dynamic_obstacles.append(obstacle)
                    spawned_positions.add(candidate)
                    debug_log(f"Spawned {obs_type} at {candidate}")
                    break
                attempts += 1
        
        self.dynamic_obstacles_spawned = True
        debug_log(f"Spawned {len(self.dynamic_obstacles)} dynamic obstacles ({obstacle_types.count('worker')} workers, {obstacle_types.count('forklift')} forklifts)")
    
    def update_dynamic_obstacles(self, current_time):
        """Update all dynamic obstacles."""
        for obstacle in self.dynamic_obstacles:
            obstacle.update(current_time, other_obstacles=self.dynamic_obstacles)
    
    def spawn_red_box(self):
        """
        Spawn a red box (high-priority cargo) at a random free position (map5)
        or at the next static location (map7).
        Only one red box can exist at a time.
        Active on map5 and map7.
        """
        # Red box feature only active on map5 and map7
        if self.map_name not in ['map5', 'map7']:
            return
        
        # For map7, use static locations
        if self.map_name == 'map7':
            # Check if we've reached the limit of 10 red boxes
            if self.total_red_boxes_spawned >= 10:
                debug_log("Red box limit reached (10 boxes). No more red boxes will spawn.")
                return
            
            # Check if we have more static locations to use
            if self.red_box_spawn_index >= len(self.static_red_box_locations):
                debug_log("All static red box locations used. No more red boxes will spawn.")
                return
            
            # If red box already exists, clear it first
            if self.red_box is not None:
                debug_log(f"Clearing existing red box at {self.red_box} before spawning new one")
                self.clear_red_box()
            
            # Spawn at the next static location
            next_location = self.static_red_box_locations[self.red_box_spawn_index]
            self.red_box = next_location
            self.red_box_spawn_time = pygame.time.get_ticks()
            self.red_box_spawn_index += 1
            self.total_red_boxes_spawned += 1
            debug_log(f"RED BOX SPAWNED at {self.red_box} (static location {self.red_box_spawn_index}/{len(self.static_red_box_locations)}) - Priority cargo active!")
            return
        
        import random
        
        # If red box already exists, clear it first
        if self.red_box is not None:
            debug_log(f"Clearing existing red box at {self.red_box} before spawning new one")
            self.clear_red_box()
        
        free_cells = self.get_free_cells()
        
        # Remove discharge dock, loading dock, and goals from spawn locations
        excluded_positions = {self.discharge_dock, self.loading_dock}
        if self.goals:
            excluded_positions.update(self.goals)
        
        # Also exclude cells within 2 cells of discharge dock (Manhattan distance)
        if self.discharge_dock:
            dock_x, dock_y = self.discharge_dock
            for cell in free_cells[:]:
                cell_x, cell_y = cell
                distance = abs(cell_x - dock_x) + abs(cell_y - dock_y)
                if distance <= 2:
                    excluded_positions.add(cell)
        
        free_cells = [cell for cell in free_cells if cell not in excluded_positions]
        
        if not free_cells:
            debug_log("Warning: No free cells available for red box spawn")
            return
        
        # Select random position
        max_attempts = 50
        for attempt in range(max_attempts):
            candidate = random.choice(free_cells)
            
            # Check if reachable from discharge dock
            if self.is_reachable(self.discharge_dock, candidate):
                self.red_box = candidate
                self.red_box_spawn_time = pygame.time.get_ticks()
                debug_log(f"RED BOX SPAWNED at {self.red_box} - Priority cargo active!")
                return
        
        # If no reachable position found, spawn anyway
        self.red_box = random.choice(free_cells)
        self.red_box_spawn_time = pygame.time.get_ticks()
        debug_log(f"RED BOX SPAWNED at {self.red_box} (may be unreachable)")
    
    def clear_red_box(self):
        """Clear the current red box."""
        if self.red_box is not None:
            debug_log(f"Red box at {self.red_box} cleared")
            self.red_box = None
            self.red_box_spawn_time = None
    
    def check_red_box_expiration(self, current_time):
        """
        Check if red box has expired (9 seconds) and clear it if so.
        Active on map5 and map7.
        
        Args:
            current_time: Current timestamp in milliseconds
            
        Returns:
            bool: True if red box expired, False otherwise
        """
        # Red box feature only active on map5 and map7
        if self.map_name not in ['map5', 'map7']:
            return False
        
        if self.red_box is not None and self.red_box_spawn_time is not None:
            time_elapsed = current_time - self.red_box_spawn_time
            if time_elapsed >= self.red_box_expiration_time:
                debug_log(f"RED BOX EXPIRED at {self.red_box} after {time_elapsed/1000:.1f}s")
                self.clear_red_box()
                return True
        return False
    
    def draw(self, surface, robot):
        """Draw the warehouse grid, obstacles, goals, and docks."""
        from constants import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, DARK_GRAY
        
        # Performance optimization: only redraw background when exploration state changes
        current_explored = robot.ogm.explored_cells if robot.ogm else set()
        needs_redraw = (self.background_surface is None or 
                        current_explored != self.last_explored_cells)
        
        if needs_redraw:
            # Create/recreate background surface
            self.background_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # Draw background - dark for unexplored, light for explored
            for y in range(WAREHOUSE_HEIGHT):
                for x in range(WAREHOUSE_WIDTH):
                    cell_x = x * GRID_SIZE
                    cell_y = y * GRID_SIZE
                    # Dark gray for unexplored cells
                    if robot.ogm and not robot.ogm.is_explored(x, y):
                        pygame.draw.rect(self.background_surface, DARK_GRAY, 
                                       (cell_x, cell_y, GRID_SIZE, GRID_SIZE))
                    else:
                        # Light/white for explored cells
                        pygame.draw.rect(self.background_surface, WHITE, 
                                       (cell_x, cell_y, GRID_SIZE, GRID_SIZE))
            
            # Draw grid lines (static)
            for x in range(0, SCREEN_WIDTH, GRID_SIZE):
                pygame.draw.line(self.background_surface, GRAY, (x, 0), (x, SCREEN_HEIGHT), 1)
            for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
                pygame.draw.line(self.background_surface, GRAY, (0, y), (SCREEN_WIDTH, y), 1)
            
            self.last_explored_cells = current_explored.copy()
        
        # Blit cached background (very fast!)
        surface.blit(self.background_surface, (0, 0))
        
        # Draw loading dock
        if self.loading_dock:
            x = self.loading_dock[0] * GRID_SIZE
            y = self.loading_dock[1] * GRID_SIZE
            pygame.draw.rect(surface, LOADING_DOCK, (x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2))
        
        # Draw discharge dock (highlight if robot has cargo)
        if self.discharge_dock:
            x = self.discharge_dock[0] * GRID_SIZE
            y = self.discharge_dock[1] * GRID_SIZE
            from constants import RED
            color = (255, 100, 100) if robot.has_cargo else DISCHARGE_DOCK
            pygame.draw.rect(surface, color, (x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2))
            if robot.has_cargo:
                pygame.draw.rect(surface, RED, (x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2), 3)
        
        # Draw goal points (packages) with priority numbers
        # Only draw goals that have been discovered (in OGM)
        font_small = pygame.font.Font(None, 18)
        discovered_goals = set()
        if robot.ogm:
            # Get discovered goals from OGM (convert set to sorted list for consistent ordering)
            discovered_goals = robot.ogm.goals
        
        for idx, (col, row) in enumerate(self.goals):
            # Only draw if goal has been discovered
            if (col, row) in discovered_goals:
                x = col * GRID_SIZE
                y = row * GRID_SIZE
                pygame.draw.rect(surface, GOAL_COLOR, (x + 4, y + 4, GRID_SIZE - 8, GRID_SIZE - 8))
                # Draw priority number
                priority_text = font_small.render(str(idx + 1), True, BLACK)
                text_rect = priority_text.get_rect(center=(x + GRID_SIZE // 2, y + GRID_SIZE // 2))
                surface.blit(priority_text, text_rect)
        
        # Draw red box (high-priority cargo) with pulsing effect (only on map5 and map7)
        if self.red_box is not None and self.map_name in ['map5', 'map7']:
            from constants import RED_BOX_COLOR
            col, row = self.red_box
            x = col * GRID_SIZE
            y = row * GRID_SIZE
            
            # Pulsing effect based on time
            current_time = pygame.time.get_ticks()
            pulse = abs(math.sin(current_time / 200.0))  # Pulse period ~400ms
            size_offset = int(3 * pulse)
            
            # Draw pulsing red box
            pygame.draw.rect(surface, RED_BOX_COLOR, 
                           (x + 2 - size_offset, y + 2 - size_offset, 
                            GRID_SIZE - 4 + 2*size_offset, GRID_SIZE - 4 + 2*size_offset))
            
            # Draw thick border for emphasis
            pygame.draw.rect(surface, (200, 0, 0), 
                           (x + 2 - size_offset, y + 2 - size_offset, 
                            GRID_SIZE - 4 + 2*size_offset, GRID_SIZE - 4 + 2*size_offset), 3)
            
            # Draw "!" symbol in white
            font_priority = pygame.font.Font(None, 32)
            priority_text = font_priority.render("!", True, WHITE)
            text_rect = priority_text.get_rect(center=(x + GRID_SIZE // 2, y + GRID_SIZE // 2))
            surface.blit(priority_text, text_rect)
        
        # Draw obstacles - only draw discovered obstacles (in OGM)
        discovered_obstacles = set()
        if robot.ogm:
            discovered_obstacles = robot.ogm.obstacles
        
        for col, row in self.obstacles:
            # Only draw if obstacle has been discovered
            if (col, row) in discovered_obstacles:
                x = col * GRID_SIZE
                y = row * GRID_SIZE
                pygame.draw.rect(surface, YELLOW, (x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2))
                pygame.draw.rect(surface, DARK_GRAY, (x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2), 1)
        
        # Draw dynamic obstacles with trajectories and velocity vectors
        if robot and robot.local_mapper:
            # Get dynamic obstacles in local map radius
            robot_pos = (robot.x, robot.y)
            dynamic_obstacles_info = robot.local_mapper.get_dynamic_obstacles_in_radius(robot_pos, radius=robot.local_mapper.radius)
            
            for obstacle_id, current_pos, predicted_pos in dynamic_obstacles_info:
                # Draw velocity vector (arrow from current to predicted position)
                if predicted_pos:
                    curr_x = int(current_pos[0] * GRID_SIZE + GRID_SIZE // 2)
                    curr_y = int(current_pos[1] * GRID_SIZE + GRID_SIZE // 2)
                    pred_x = int(predicted_pos[0] * GRID_SIZE + GRID_SIZE // 2)
                    pred_y = int(predicted_pos[1] * GRID_SIZE + GRID_SIZE // 2)
                    
                    # Draw predicted trajectory line (dotted)
                    pygame.draw.line(surface, (255, 200, 0), (curr_x, curr_y), (pred_x, pred_y), 2)
                    
                    # Draw arrow head
                    if abs(pred_x - curr_x) > 2 or abs(pred_y - curr_y) > 2:
                        angle = math.atan2(pred_y - curr_y, pred_x - curr_x)
                        arrow_size = 6
                        head_x1 = pred_x - arrow_size * math.cos(angle - math.pi / 6)
                        head_y1 = pred_y - arrow_size * math.sin(angle - math.pi / 6)
                        head_x2 = pred_x - arrow_size * math.cos(angle + math.pi / 6)
                        head_y2 = pred_y - arrow_size * math.sin(angle + math.pi / 6)
                        pygame.draw.polygon(surface, (255, 200, 0), 
                                          [(pred_x, pred_y), (int(head_x1), int(head_y1)), (int(head_x2), int(head_y2))])
        
        # Draw dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            obstacle.draw(surface)
