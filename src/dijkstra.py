"""
Dijkstra pathfinding algorithm for warehouse robot simulation.
Implements Dijkstra's algorithm for grid-based pathfinding without heuristic.
"""

import heapq
from constants import WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT
from ogm import FREE, GOAL, OCCUPIED, UNKNOWN

DEBUG = True

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[DIJKSTRA DEBUG] {message}")


def neighbors4(x, y):
    """Get 4-neighbors of a cell in fixed order (N, E, S, W)."""
    # Fixed order: N, E, S, W (up, right, down, left)
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < WAREHOUSE_WIDTH and 0 <= ny < WAREHOUSE_HEIGHT:
            yield (nx, ny)


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


def dijkstra(start, target, ogm, allow_goals=True):
    """
    Dijkstra's algorithm pathfinding from start to target using the occupancy grid map.
    
    Args:
        start: (x, y) starting position in grid coordinates
        target: (x, y) target position in grid coordinates
        ogm: OccupancyGridMap to check obstacles
        allow_goals: If True, GOAL cells are traversable (default: True)
        
    Returns:
        list: Path as list of (x, y) tuples, or None if unreachable
    """
    if start == target:
        debug_print(f"Dijkstra start equals target: {start}")
        return [start]
    
    sx, sy = start
    tx, ty = target
    
    debug_print(f"Dijkstra planning from ({sx}, {sy}) to ({tx}, {ty})")
    
    # Check if positions are valid and traversable
    if not is_traversable(sx, sy, ogm, allow_goals):
        debug_print(f"Dijkstra start position ({sx}, {sy}) is not traversable")
        start_state = ogm.get_cell_state(sx, sy) if (0 <= sx < WAREHOUSE_WIDTH and 0 <= sy < WAREHOUSE_HEIGHT) else "OUT_OF_BOUNDS"
        debug_print(f"  Start cell state: {start_state}")
        if start_state == UNKNOWN:
            debug_print(f"  ERROR: Start position is UNKNOWN - exploration incomplete!")
        return None
    
    if not is_traversable(tx, ty, ogm, allow_goals):
        debug_print(f"Dijkstra target position ({tx}, {ty}) is not traversable")
        target_state = ogm.get_cell_state(tx, ty) if (0 <= tx < WAREHOUSE_WIDTH and 0 <= ty < WAREHOUSE_HEIGHT) else "OUT_OF_BOUNDS"
        debug_print(f"  Target cell state: {target_state}")
        if target_state == UNKNOWN:
            debug_print(f"  ERROR: Target position is UNKNOWN - exploration incomplete!")
        return None
    
    # Dijkstra's algorithm using priority queue
    open_set = []  # Priority queue: (distance, x, y)
    heapq.heappush(open_set, (0, sx, sy))
    came_from = {}
    distance = {(sx, sy): 0}
    visited = set()
    
    iterations = 0
    max_iterations = WAREHOUSE_WIDTH * WAREHOUSE_HEIGHT * 2  # Safety limit
    
    while open_set and iterations < max_iterations:
        iterations += 1
        
        # Pop the node with minimum distance
        current_dist, cx, cy = heapq.heappop(open_set)
        
        # Skip if already processed
        if (cx, cy) in visited:
            continue
        
        visited.add((cx, cy))
        
        # Check if reached target
        if (cx, cy) == (tx, ty):
            # Reconstruct path
            path = []
            node = (cx, cy)
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            path.reverse()
            debug_print(f"Dijkstra path found: {len(path)} steps from ({sx}, {sy}) to ({tx}, {ty}) in {iterations} iterations")
            return path
        
        # Check neighbors
        for nx, ny in neighbors4(cx, cy):
            # Skip if already processed
            if (nx, ny) in visited:
                continue
            
            # Check if neighbor is traversable
            if not is_traversable(nx, ny, ogm, allow_goals):
                continue
            
            # Calculate tentative distance (all edges have weight 1)
            tentative_dist = distance[(cx, cy)] + 1
            
            # Update if this is a shorter path
            if (nx, ny) not in distance or tentative_dist < distance[(nx, ny)]:
                came_from[(nx, ny)] = (cx, cy)
                distance[(nx, ny)] = tentative_dist
                heapq.heappush(open_set, (tentative_dist, nx, ny))
    
    if iterations >= max_iterations:
        debug_print(f"Dijkstra pathfinding exceeded max iterations ({max_iterations})")
    
    debug_print(f"Dijkstra path not found from ({sx}, {sy}) to ({tx}, {ty}) after {iterations} iterations")
    return None  # No path found

