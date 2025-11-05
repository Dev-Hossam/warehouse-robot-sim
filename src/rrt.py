"""
RRT (Rapidly-exploring Random Tree) pathfinding algorithm for warehouse robot simulation.
Implements RRT for grid-based pathfinding with sampling-based exploration.
"""

import random
import math
from constants import WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT
from ogm import FREE, GOAL, OCCUPIED, UNKNOWN

DEBUG = True

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[RRT DEBUG] {message}")


def neighbors4(x, y):
    """Get 4-neighbors of a cell in fixed order (N, E, S, W)."""
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < WAREHOUSE_WIDTH and 0 <= ny < WAREHOUSE_HEIGHT:
            yield (nx, ny)


def euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


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


def find_nearest_node(tree, x, y):
    """Find the nearest node in the tree to the given point."""
    min_dist = float('inf')
    nearest = None
    
    for node in tree:
        dist = euclidean_distance(node[0], node[1], x, y)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    
    return nearest


def step_towards(x1, y1, x2, y2, step_size=2):
    """Step from (x1, y1) towards (x2, y2) by step_size using grid-based movement."""
    dist = euclidean_distance(x1, y1, x2, y2)
    if dist <= step_size:
        return (int(x2), int(y2))
    
    # Move towards target by step_size using grid-based movement (4-connected)
    # Use Manhattan-like movement to ensure we stay on grid
    dx = x2 - x1
    dy = y2 - y1
    
    # Normalize to step_size
    if abs(dx) + abs(dy) == 0:
        return (int(x1), int(y1))
    
    # Move in the direction with larger component first
    if abs(dx) > abs(dy):
        # Move horizontally first
        step_x = step_size if dx > 0 else -step_size
        new_x = x1 + step_x
        new_y = y1
    else:
        # Move vertically first
        step_y = step_size if dy > 0 else -step_size
        new_x = x1
        new_y = y1 + step_y
    
    # Ensure we don't overshoot
    if abs(new_x - x2) + abs(new_y - y2) < abs(x1 - x2) + abs(y1 - y2):
        return (int(new_x), int(new_y))
    else:
        return (int(x2), int(y2))


def rrt(start, target, ogm, allow_goals=True, max_iterations=5000, step_size=2, goal_bias=0.1):
    """
    RRT pathfinding from start to target using the occupancy grid map.
    
    Args:
        start: (x, y) starting position in grid coordinates
        target: (x, y) target position in grid coordinates
        ogm: OccupancyGridMap to check obstacles
        allow_goals: If True, GOAL cells are traversable (default: True)
        max_iterations: Maximum number of iterations (default: 5000)
        step_size: Step size for tree expansion (default: 2)
        goal_bias: Probability of sampling goal instead of random (default: 0.1)
        
    Returns:
        list: Path as list of (x, y) tuples, or None if unreachable
    """
    if start == target:
        debug_print(f"RRT start equals target: {start}")
        return [start]
    
    sx, sy = start
    tx, ty = target
    
    debug_print(f"RRT planning from ({sx}, {sy}) to ({tx}, {ty})")
    
    # Check if positions are valid and traversable
    if not is_traversable(sx, sy, ogm, allow_goals):
        debug_print(f"RRT start position ({sx}, {sy}) is not traversable")
        return None
    
    if not is_traversable(tx, ty, ogm, allow_goals):
        debug_print(f"RRT target position ({tx}, {ty}) is not traversable")
        return None
    
    # Initialize RRT tree
    tree = {}  # {node: parent}
    tree[start] = None
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        
        # Sample random point (with goal bias)
        if random.random() < goal_bias:
            # Sample goal
            rand_x, rand_y = tx, ty
        else:
            # Sample random point in free space
            rand_x = random.randint(0, WAREHOUSE_WIDTH - 1)
            rand_y = random.randint(0, WAREHOUSE_HEIGHT - 1)
        
        # Find nearest node in tree
        nearest = find_nearest_node(tree.keys(), rand_x, rand_y)
        if nearest is None:
            continue
        
        # Step towards random point
        new_x, new_y = step_towards(nearest[0], nearest[1], rand_x, rand_y, step_size)
        
        # Check if new point is valid and traversable
        if not is_traversable(new_x, new_y, ogm, allow_goals):
            continue
        
        # Check if path from nearest to new point is clear (grid-based check)
        if not is_path_clear_grid(nearest[0], nearest[1], new_x, new_y, ogm, allow_goals):
            continue
        
        # Add new node to tree
        new_node = (new_x, new_y)
        tree[new_node] = nearest
        
        # Check if we're close enough to target
        if euclidean_distance(new_x, new_y, tx, ty) <= step_size * 1.5:
            # Check if direct path to target is clear
            if is_path_clear_grid(new_x, new_y, tx, ty, ogm, allow_goals):
                # Reconstruct path from start to target
                path = []
                node = (tx, ty)
                path.append(node)
                
                # Trace back to start
                current = new_node
                while current is not None:
                    path.append(current)
                    current = tree.get(current)
                
                path.reverse()
                
                # Smooth the path to reduce unnecessary waypoints
                path = smooth_path(path, ogm, allow_goals)
                
                debug_print(f"RRT path found: {len(path)} steps from ({sx}, {sy}) to ({tx}, {ty}) in {iterations} iterations")
                return path
    
    debug_print(f"RRT path not found from ({sx}, {sy}) to ({tx}, {ty}) after {iterations} iterations")
    return None  # No path found


def is_path_clear_grid(x1, y1, x2, y2, ogm, allow_goals=True):
    """Check if grid-based path between two points is clear (4-connected grid path)."""
    # Use grid-based path (4-connected neighbors only)
    x, y = x1, y1
    while x != x2 or y != y2:
        if not is_traversable(x, y, ogm, allow_goals):
            return False
        
        # Move one step towards target (grid-based)
        if x < x2:
            x += 1
        elif x > x2:
            x -= 1
        elif y < y2:
            y += 1
        elif y > y2:
            y -= 1
        else:
            break
    
    # Check final cell
    return is_traversable(x2, y2, ogm, allow_goals)


def smooth_path(path, ogm, allow_goals=True):
    """Smooth RRT path by removing unnecessary waypoints."""
    if len(path) <= 2:
        return path
    
    smoothed = [path[0]]
    i = 0
    
    while i < len(path) - 1:
        # Try to connect current point to points further ahead
        j = len(path) - 1
        while j > i + 1:
            if is_path_clear_grid(path[i][0], path[i][1], path[j][0], path[j][1], ogm, allow_goals):
                # Can skip intermediate points
                smoothed.append(path[j])
                i = j
                break
            j -= 1
        
        if j == i + 1:
            # Can't skip, add next point
            smoothed.append(path[i + 1])
            i += 1
    
    return smoothed

