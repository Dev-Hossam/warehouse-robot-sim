"""
A* pathfinding algorithm for warehouse robot simulation.
Implements A* with Manhattan heuristic for grid-based pathfinding.
"""

import heapq
from constants import WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT
from ogm import FREE, GOAL, OCCUPIED, UNKNOWN

DEBUG = True

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[ASTAR DEBUG] {message}")


def neighbors4(x, y):
    """Get 4-neighbors of a cell in fixed order (N, E, S, W)."""
    # Fixed order: N, E, S, W (up, right, down, left)
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < WAREHOUSE_WIDTH and 0 <= ny < WAREHOUSE_HEIGHT:
            yield (nx, ny)


def manhattan_distance(x1, y1, x2, y2):
    """Calculate Manhattan distance between two points."""
    return abs(x2 - x1) + abs(y2 - y1)


def is_traversable(x, y, ogm, allow_goals=True):
    """Check if a cell is traversable for pathfinding."""
    if not (0 <= x < WAREHOUSE_WIDTH and 0 <= y < WAREHOUSE_HEIGHT):
        return False
    
    cell_state = ogm.get_cell_state(x, y)
    
    # OCCUPIED cells are never traversable
    if cell_state == OCCUPIED:
        return False
    
    # UNKNOWN cells are not traversable (should be explored first)
    if cell_state == UNKNOWN:
        return False
    
    # FREE cells are always traversable
    if cell_state == FREE:
        return True
    
    # GOAL cells are traversable if allowed
    if cell_state == GOAL:
        return allow_goals
    
    return False


def astar(start, target, ogm, allow_goals=True):
    """
    A* pathfinding from start to target using the occupancy grid map.
    
    Args:
        start: (x, y) starting position in grid coordinates
        target: (x, y) target position in grid coordinates
        ogm: OccupancyGridMap to check obstacles
        allow_goals: If True, GOAL cells are traversable (default: True)
        
    Returns:
        list: Path as list of (x, y) tuples, or None if unreachable
    """
    if start == target:
        debug_print(f"A* start equals target: {start}")
        return [start]
    
    sx, sy = start
    tx, ty = target
    
    debug_print(f"A* planning from ({sx}, {sy}) to ({tx}, {ty})")
    
    # Check if positions are valid and traversable
    if not is_traversable(sx, sy, ogm, allow_goals):
        debug_print(f"A* start position ({sx}, {sy}) is not traversable")
        start_state = ogm.get_cell_state(sx, sy) if (0 <= sx < WAREHOUSE_WIDTH and 0 <= sy < WAREHOUSE_HEIGHT) else "OUT_OF_BOUNDS"
        debug_print(f"  Start cell state: {start_state}")
        if start_state == UNKNOWN:
            debug_print(f"  ERROR: Start position is UNKNOWN - exploration incomplete!")
        return None
    
    if not is_traversable(tx, ty, ogm, allow_goals):
        debug_print(f"A* target position ({tx}, {ty}) is not traversable")
        target_state = ogm.get_cell_state(tx, ty) if (0 <= tx < WAREHOUSE_WIDTH and 0 <= ty < WAREHOUSE_HEIGHT) else "OUT_OF_BOUNDS"
        debug_print(f"  Target cell state: {target_state}")
        if target_state == UNKNOWN:
            debug_print(f"  ERROR: Target position is UNKNOWN - exploration incomplete!")
        return None
    
    # A* algorithm using heapq priority queue (O(log n) instead of O(n log n))
    open_set = []  # Priority queue: (f_score, x, y)
    heapq.heappush(open_set, (0, sx, sy))
    came_from = {}
    g_score = {(sx, sy): 0}
    h = manhattan_distance(sx, sy, tx, ty)
    f_score = {(sx, sy): h}
    closed_set = set()
    
    iterations = 0
    max_iterations = WAREHOUSE_WIDTH * WAREHOUSE_HEIGHT * 2  # Safety limit
    
    while open_set and iterations < max_iterations:
        iterations += 1
        
        # Pop the best node from priority queue (O(log n))
        current_f, cx, cy = heapq.heappop(open_set)
        
        # Skip if already processed
        if (cx, cy) in closed_set:
            continue
        
        closed_set.add((cx, cy))
        
        if (cx, cy) == (tx, ty):
            # Reconstruct path
            path = []
            node = (cx, cy)
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            path.reverse()
            debug_print(f"A* path found: {len(path)} steps from ({sx}, {sy}) to ({tx}, {ty}) in {iterations} iterations")
            return path
        
        # Check neighbors
        for nx, ny in neighbors4(cx, cy):
            # Skip if already processed
            if (nx, ny) in closed_set:
                continue
            
            # Check if neighbor is traversable
            if not is_traversable(nx, ny, ogm, allow_goals):
                continue
            
            # Calculate tentative g-score
            tentative_g = g_score[(cx, cy)] + 1
            
            # Update if this is a better path
            if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                came_from[(nx, ny)] = (cx, cy)
                g_score[(nx, ny)] = tentative_g
                h = manhattan_distance(nx, ny, tx, ty)
                f = tentative_g + h
                f_score[(nx, ny)] = f
                
                # Add to open set priority queue if not already there
                # Check if already in open_set (we'll add it anyway, duplicate check happens in closed_set)
                heapq.heappush(open_set, (f, nx, ny))
    
    if iterations >= max_iterations:
        debug_print(f"A* pathfinding exceeded max iterations ({max_iterations})")
    
    debug_print(f"A* path not found from ({sx}, {sy}) to ({tx}, {ty}) after {iterations} iterations")
    debug_print(f"  Closed set size: {len(closed_set)}")
    debug_print(f"  Open set size: {len(open_set)}")
    
    # Debug: Check if start and target are reachable
    start_state = ogm.get_cell_state(sx, sy)
    target_state = ogm.get_cell_state(tx, ty)
    debug_print(f"  Start state: {start_state}, Target state: {target_state}")
    
    # Check for UNKNOWN cells blocking the path
    unknown_count = 0
    unknown_cells = []
    # Sample cells along direct path to check for UNKNOWN cells
    dx = tx - sx
    dy = ty - sy
    steps = max(abs(dx), abs(dy))
    if steps > 0:
        for i in range(steps + 1):
            check_x = sx + (dx * i) // steps
            check_y = sy + (dy * i) // steps
            if 0 <= check_x < WAREHOUSE_WIDTH and 0 <= check_y < WAREHOUSE_HEIGHT:
                cell_state = ogm.get_cell_state(check_x, check_y)
                if cell_state == UNKNOWN:
                    unknown_count += 1
                    unknown_cells.append((check_x, check_y))
    
    if unknown_count > 0:
        debug_print(f"  WARNING: Path blocked by {unknown_count} UNKNOWN cell(s)")
        debug_print(f"  UNKNOWN cells detected: {unknown_cells[:5]}...")  # Show first 5
        debug_print(f"  This suggests exploration is incomplete. UNKNOWN cells are not traversable.")
    else:
        # Check neighbors of explored cells for UNKNOWN
        unknown_neighbors = 0
        for x, y in list(closed_set)[:10]:  # Sample first 10 explored cells
            for nx, ny in neighbors4(x, y):
                if ogm.get_cell_state(nx, ny) == UNKNOWN:
                    unknown_neighbors += 1
        if unknown_neighbors > 0:
            debug_print(f"  WARNING: {unknown_neighbors} UNKNOWN cells adjacent to explored area")
            debug_print(f"  This suggests exploration may be incomplete.")
    
    return None  # No path found


def plan_multi_goal_path(start, goals, discharge_dock, ogm):
    """
    Plan optimal path to visit all goals in priority order and deliver to discharge dock.
    Uses nearest-neighbor heuristic for goal ordering.
    
    Args:
        start: (x, y) starting position
        goals: List of (x, y) goal positions in priority order
        discharge_dock: (x, y) discharge dock position
        ogm: OccupancyGridMap to check obstacles
        
    Returns:
        list: Complete path as list of (x, y) tuples
    """
    if not goals:
        # No goals, just return to discharge dock
        path = astar(start, discharge_dock, ogm)
        return path if path else []
    
    complete_path = []
    current_pos = start
    remaining_goals = list(goals)
    
    while remaining_goals:
        # Find nearest goal from current position
        nearest_goal = None
        min_dist = float('inf')
        
        for goal in remaining_goals:
            dist = manhattan_distance(current_pos[0], current_pos[1], goal[0], goal[1])
            if dist < min_dist:
                min_dist = dist
                nearest_goal = goal
        
        if nearest_goal is None:
            break
        
        # Plan path to nearest goal
        path_to_goal = astar(current_pos, nearest_goal, ogm)
        if not path_to_goal:
            debug_print(f"Warning: No path to goal {nearest_goal}, skipping")
            remaining_goals.remove(nearest_goal)
            continue
        
        # Add path to goal (skip first point - it's current position)
        if len(path_to_goal) > 1:
            complete_path.extend(path_to_goal[1:])
        current_pos = nearest_goal
        
        # Remove goal from list
        remaining_goals.remove(nearest_goal)
        
        # Plan path from goal to discharge dock
        path_to_dock = astar(current_pos, discharge_dock, ogm)
        if not path_to_dock:
            debug_print(f"Warning: No path from goal {nearest_goal} to discharge dock")
            continue
        
        # Add path to dock (skip first point - it's current position)
        if len(path_to_dock) > 1:
            complete_path.extend(path_to_dock[1:])
        current_pos = discharge_dock
    
    debug_print(f"Multi-goal path planned: {len(complete_path)} steps")
    return complete_path

