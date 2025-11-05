"""
PRM (Probabilistic Roadmap) pathfinding algorithm for warehouse robot simulation.
Implements PRM for grid-based pathfinding with sampling-based roadmap construction.
"""

import random
import math
import heapq
from constants import WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT
from ogm import FREE, GOAL, OCCUPIED, UNKNOWN

DEBUG = True

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[PRM DEBUG] {message}")


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


def is_path_clear(x1, y1, x2, y2, ogm, allow_goals=True):
    """Check if path between two points is clear (simple line check)."""
    # Use Bresenham-like line check
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    x, y = x1, y1
    while True:
        if not is_traversable(x, y, ogm, allow_goals):
            return False
        
        if x == x2 and y == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return True


def build_roadmap(ogm, allow_goals=True, num_samples=500, connection_radius=5):
    """
    Build a probabilistic roadmap by sampling free space and connecting nearby nodes.
    
    Args:
        ogm: OccupancyGridMap to check obstacles
        allow_goals: If True, GOAL cells are traversable
        num_samples: Number of samples to generate
        connection_radius: Maximum distance to connect nodes
        
    Returns:
        dict: Roadmap graph {node: [neighbors]}
    """
    roadmap = {}
    samples = []
    
    # Sample free space
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        x = random.randint(0, WAREHOUSE_WIDTH - 1)
        y = random.randint(0, WAREHOUSE_HEIGHT - 1)
        
        if is_traversable(x, y, ogm, allow_goals):
            samples.append((x, y))
            roadmap[(x, y)] = []
    
    debug_print(f"PRM: Generated {len(samples)} samples")
    
    # Connect nearby nodes
    for i, node1 in enumerate(samples):
        for node2 in samples[i+1:]:
            dist = euclidean_distance(node1[0], node1[1], node2[0], node2[1])
            if dist <= connection_radius:
                # Check if path is clear
                if is_path_clear(node1[0], node1[1], node2[0], node2[1], ogm, allow_goals):
                    roadmap[node1].append(node2)
                    roadmap[node2].append(node1)
    
    debug_print(f"PRM: Built roadmap with {len(roadmap)} nodes")
    return roadmap


def find_nearest_roadmap_node(roadmap, x, y):
    """Find the nearest node in the roadmap to the given point."""
    if not roadmap:
        return None
    
    min_dist = float('inf')
    nearest = None
    
    for node in roadmap:
        dist = euclidean_distance(node[0], node[1], x, y)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    
    return nearest


def prm(start, target, ogm, allow_goals=True, num_samples=500, connection_radius=5):
    """
    PRM pathfinding from start to target using the occupancy grid map.
    
    Args:
        start: (x, y) starting position in grid coordinates
        target: (x, y) target position in grid coordinates
        ogm: OccupancyGridMap to check obstacles
        allow_goals: If True, GOAL cells are traversable (default: True)
        num_samples: Number of samples for roadmap (default: 500)
        connection_radius: Connection radius for roadmap (default: 5)
        
    Returns:
        list: Path as list of (x, y) tuples, or None if unreachable
    """
    if start == target:
        debug_print(f"PRM start equals target: {start}")
        return [start]
    
    sx, sy = start
    tx, ty = target
    
    debug_print(f"PRM planning from ({sx}, {sy}) to ({tx}, {ty})")
    
    # Check if positions are valid and traversable
    if not is_traversable(sx, sy, ogm, allow_goals):
        debug_print(f"PRM start position ({sx}, {sy}) is not traversable")
        return None
    
    if not is_traversable(tx, ty, ogm, allow_goals):
        debug_print(f"PRM target position ({tx}, {ty}) is not traversable")
        return None
    
    # Build or reuse roadmap (could be cached, but for now rebuild each time)
    roadmap = build_roadmap(ogm, allow_goals, num_samples, connection_radius)
    
    # Find nearest roadmap nodes to start and target
    start_node = find_nearest_roadmap_node(roadmap, sx, sy)
    target_node = find_nearest_roadmap_node(roadmap, tx, ty)
    
    if start_node is None or target_node is None:
        debug_print("PRM: Could not find nearest roadmap nodes")
        return None
    
    # Check if we can connect start to roadmap
    if not is_path_clear(sx, sy, start_node[0], start_node[1], ogm, allow_goals):
        # Try adding start to roadmap
        roadmap[start] = []
        if is_path_clear(sx, sy, start_node[0], start_node[1], ogm, allow_goals):
            roadmap[start].append(start_node)
            roadmap[start_node].append(start)
            start_node = start
        else:
            debug_print("PRM: Cannot connect start to roadmap")
            return None
    
    # Check if we can connect target to roadmap
    if not is_path_clear(tx, ty, target_node[0], target_node[1], ogm, allow_goals):
        # Try adding target to roadmap
        roadmap[target] = []
        if is_path_clear(tx, ty, target_node[0], target_node[1], ogm, allow_goals):
            roadmap[target].append(target_node)
            roadmap[target_node].append(target)
            target_node = target
        else:
            debug_print("PRM: Cannot connect target to roadmap")
            return None
    
    # Use Dijkstra to find path through roadmap
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    came_from = {}
    distance = {start_node: 0}
    visited = set()
    
    while open_set:
        current_dist, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        if current == target_node:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            path.reverse()
            
            # Add start and target if not already in path
            if path[0] != start:
                path.insert(0, start)
            if path[-1] != target:
                path.append(target)
            
            debug_print(f"PRM path found: {len(path)} steps from ({sx}, {sy}) to ({tx}, {ty})")
            return path
        
        # Explore neighbors
        for neighbor in roadmap.get(current, []):
            if neighbor in visited:
                continue
            
            dist = euclidean_distance(current[0], current[1], neighbor[0], neighbor[1])
            tentative_dist = distance[current] + dist
            
            if neighbor not in distance or tentative_dist < distance[neighbor]:
                distance[neighbor] = tentative_dist
                came_from[neighbor] = current
                heapq.heappush(open_set, (tentative_dist, neighbor))
    
    debug_print(f"PRM path not found from ({sx}, {sy}) to ({tx}, {ty})")
    return None  # No path found

