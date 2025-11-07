"""
Local Mapper module for warehouse robot simulation.
Implements high-resolution local mapping with dynamic obstacle tracking and temporal decay.
"""

import math
import time
from collections import defaultdict, deque
from constants import WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT, debug_log

DEBUG = True

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[LOCAL_MAPPER DEBUG] {message}")

# Cell states for local map
LOCAL_UNKNOWN = -1
LOCAL_FREE = 0
LOCAL_OCCUPIED = 1
LOCAL_DYNAMIC_OCCUPIED = 2


class DynamicObstacleTracker:
    """Tracks dynamic obstacles with velocity estimation and position prediction."""
    
    def __init__(self, max_history=10):
        """
        Initialize dynamic obstacle tracker.
        
        Args:
            max_history: Maximum number of position history entries per obstacle
        """
        self.obstacle_history = defaultdict(lambda: deque(maxlen=max_history))  # {obstacle_id: [(pos, timestamp), ...]}
        self.obstacle_velocities = {}  # {obstacle_id: (vx, vy)}
        self.max_history = max_history
    
    def update_obstacle_position(self, obstacle_id, pos, timestamp):
        """
        Update obstacle position in history.
        
        Args:
            obstacle_id: Unique identifier for obstacle
            pos: (x, y) position tuple
            timestamp: Current timestamp
        """
        self.obstacle_history[obstacle_id].append((pos, timestamp))
        
        # Update velocity estimate
        if len(self.obstacle_history[obstacle_id]) >= 2:
            self.obstacle_velocities[obstacle_id] = self.estimate_velocity(obstacle_id)
    
    def estimate_velocity(self, obstacle_id):
        """
        Estimate velocity from position history.
        
        Args:
            obstacle_id: Obstacle identifier
            
        Returns:
            (vx, vy) velocity tuple in grid cells per second, or (0, 0) if insufficient data
        """
        history = self.obstacle_history[obstacle_id]
        if len(history) < 2:
            return (0.0, 0.0)
        
        # Use most recent two positions for velocity estimate
        (pos1, t1) = history[-2]
        (pos2, t2) = history[-1]
        
        dt = t2 - t1
        if dt <= 0:
            return (0.0, 0.0)
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        vx = dx / dt * 1000.0  # Convert to cells per second (assuming timestamps in ms)
        vy = dy / dt * 1000.0
        
        return (vx, vy)
    
    def predict_position(self, obstacle_id, steps_ahead=2, time_per_step=0.5):
        """
        Predict future position of obstacle.
        
        Args:
            obstacle_id: Obstacle identifier
            steps_ahead: Number of steps to predict ahead
            time_per_step: Time per step in seconds
            
        Returns:
            (x, y) predicted position, or None if insufficient data
        """
        if obstacle_id not in self.obstacle_history or len(self.obstacle_history[obstacle_id]) == 0:
            return None
        
        # Get current position
        current_pos, current_time = self.obstacle_history[obstacle_id][-1]
        
        # Get velocity
        if obstacle_id not in self.obstacle_velocities:
            return current_pos
        
        vx, vy = self.obstacle_velocities[obstacle_id]
        
        # Predict future position (linear prediction)
        prediction_time = steps_ahead * time_per_step
        predicted_x = current_pos[0] + vx * prediction_time
        predicted_y = current_pos[1] + vy * prediction_time
        
        return (predicted_x, predicted_y)
    
    def get_dynamic_obstacles_in_radius(self, robot_pos, radius):
        """
        Get all dynamic obstacles within radius of robot.
        
        Args:
            robot_pos: (x, y) robot position
            radius: Search radius in grid cells
            
        Returns:
            List of (obstacle_id, current_pos, predicted_pos) tuples
        """
        obstacles_in_range = []
        rx, ry = robot_pos
        
        for obstacle_id, history in self.obstacle_history.items():
            if len(history) == 0:
                continue
            
            current_pos, _ = history[-1]
            dist = math.sqrt((current_pos[0] - rx)**2 + (current_pos[1] - ry)**2)
            
            if dist <= radius:
                predicted_pos = self.predict_position(obstacle_id, steps_ahead=2)
                obstacles_in_range.append((obstacle_id, current_pos, predicted_pos))
        
        return obstacles_in_range
    
    def clear_old_obstacles(self, current_time, max_age=5.0):
        """
        Remove obstacles that haven't been seen for a while.
        
        Args:
            current_time: Current timestamp
            max_age: Maximum age in seconds before removing
        """
        to_remove = []
        for obstacle_id, history in self.obstacle_history.items():
            if len(history) == 0:
                to_remove.append(obstacle_id)
                continue
            
            last_time = history[-1][1]
            age = (current_time - last_time) / 1000.0  # Convert ms to seconds
            
            if age > max_age:
                to_remove.append(obstacle_id)
        
        for obstacle_id in to_remove:
            del self.obstacle_history[obstacle_id]
            if obstacle_id in self.obstacle_velocities:
                del self.obstacle_velocities[obstacle_id]


class LocalMapper:
    """
    Local mapper that maintains a high-resolution map around the robot.
    Tracks dynamic obstacles and provides temporal decay of observations.
    """
    
    def __init__(self, radius=12, decay_half_life=2.0):
        """
        Initialize local mapper.
        
        Args:
            radius: Local map radius in grid cells (default: 12)
            decay_half_life: Half-life for temporal decay in seconds (default: 2.0)
        """
        self.radius = radius
        self.decay_half_life = decay_half_life
        
        # Local map: {(x, y): {'state': state, 'timestamp': timestamp, 'confidence': confidence}}
        # Coordinates are in global grid space
        self.local_map = {}
        
        # Dynamic obstacle tracker
        self.obstacle_tracker = DynamicObstacleTracker(max_history=10)
        
        # Configuration
        self.prediction_horizon = 2  # Steps ahead to predict (increased from 1 to provide more warning time)
        self.min_confidence = 0.3  # Minimum confidence to consider observation valid
        
        debug_print(f"LocalMapper initialized with radius={radius}, decay_half_life={decay_half_life}")
    
    def update_local_map(self, robot_pos, sensor_data=None):
        """
        Update local map from sensor data.
        
        Args:
            robot_pos: (x, y) robot position in global coordinates
            sensor_data: Optional sensor data (currently unused, uses global OGM)
        """
        # This will be called from robot's update_with_sensor
        # The actual update happens via update_from_global_ogm
        pass
    
    def update_from_global_ogm(self, robot_pos, ogm, warehouse):
        """
        Update local map from global OGM within local radius.
        
        Args:
            robot_pos: (x, y) robot position
            ogm: Global OccupancyGridMap
            warehouse: Warehouse object
        """
        rx, ry = robot_pos
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Update local map from global OGM
        for y in range(max(0, int(ry) - self.radius), min(WAREHOUSE_HEIGHT, int(ry) + self.radius + 1)):
            for x in range(max(0, int(rx) - self.radius), min(WAREHOUSE_WIDTH, int(rx) + self.radius + 1)):
                # Check distance from robot
                dist = math.sqrt((x - rx)**2 + (y - ry)**2)
                if dist > self.radius:
                    continue
                
                # Get state from global OGM
                cell_state = ogm.get_cell_state(x, y)
                
                # Map global states to local states
                if cell_state == 1:  # OCCUPIED
                    self.local_map[(x, y)] = {
                        'state': LOCAL_OCCUPIED,
                        'timestamp': current_time,
                        'confidence': 1.0
                    }
                elif cell_state == 0:  # FREE
                    # Only update if not already marked as dynamic obstacle
                    if (x, y) not in self.local_map or \
                       self.local_map[(x, y)].get('state') != LOCAL_DYNAMIC_OCCUPIED:
                        self.local_map[(x, y)] = {
                            'state': LOCAL_FREE,
                            'timestamp': current_time,
                            'confidence': 1.0
                        }
                elif cell_state == 2:  # GOAL
                    # Only update if not already marked as dynamic obstacle
                    if (x, y) not in self.local_map or \
                       self.local_map[(x, y)].get('state') != LOCAL_DYNAMIC_OCCUPIED:
                        self.local_map[(x, y)] = {
                            'state': LOCAL_FREE,  # Goals are traversable
                            'timestamp': current_time,
                            'confidence': 1.0
                        }
    
    def update_dynamic_obstacles(self, dynamic_obstacles, current_time):
        """
        Update local map with dynamic obstacle positions.
        
        Args:
            dynamic_obstacles: List of DynamicObstacle objects
            current_time: Current timestamp in milliseconds
        """
        # Track which cells are currently occupied by obstacles (actual positions and predictions)
        current_obstacle_cells = set()
        
        # Update obstacle tracker
        for i, obstacle in enumerate(dynamic_obstacles):
            obstacle_id = id(obstacle)  # Use object id as unique identifier
            pos = (obstacle.x, obstacle.y)
            self.obstacle_tracker.update_obstacle_position(obstacle_id, pos, current_time)
            
            # Mark cell as dynamically occupied
            cell_x, cell_y = int(obstacle.x), int(obstacle.y)
            current_obstacle_cells.add((cell_x, cell_y))
            self.local_map[(cell_x, cell_y)] = {
                'state': LOCAL_DYNAMIC_OCCUPIED,
                'timestamp': current_time,
                'confidence': 1.0,
                'obstacle_id': obstacle_id
            }
            
            # Also mark predicted positions (only if obstacle is moving)
            velocity = self.obstacle_tracker.obstacle_velocities.get(obstacle_id, (0, 0))
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
            
            # If velocity is not yet available, use obstacle's current direction for immediate prediction
            if speed < 0.1 and hasattr(obstacle, 'current_direction') and obstacle.current_direction:
                # Estimate velocity from direction (obstacles move 1 cell per move_cooldown ms)
                dx, dy = obstacle.current_direction
                move_cooldown_sec = obstacle.move_cooldown / 1000.0  # Convert to seconds
                if move_cooldown_sec > 0:
                    estimated_vx = dx / move_cooldown_sec  # cells per second
                    estimated_vy = dy / move_cooldown_sec
                    velocity = (estimated_vx, estimated_vy)
                    speed = math.sqrt(estimated_vx**2 + estimated_vy**2)
                    # Store this estimate for immediate use
                    self.obstacle_tracker.obstacle_velocities[obstacle_id] = velocity
            
            # Predict if obstacle is moving (using either calculated or estimated velocity)
            if speed > 0.1:
                # Use actual obstacle move cooldown (0.25s) for accurate prediction timing
                predicted_pos = self.obstacle_tracker.predict_position(obstacle_id, steps_ahead=self.prediction_horizon, time_per_step=0.25)
                if predicted_pos:
                    pred_x, pred_y = int(predicted_pos[0]), int(predicted_pos[1])
                    if 0 <= pred_x < WAREHOUSE_WIDTH and 0 <= pred_y < WAREHOUSE_HEIGHT:
                        current_obstacle_cells.add((pred_x, pred_y))
                        # Lower confidence for predictions - only mark if not already a static obstacle
                        if (pred_x, pred_y) not in self.local_map or \
                           self.local_map[(pred_x, pred_y)].get('state') != LOCAL_OCCUPIED:
                            self.local_map[(pred_x, pred_y)] = {
                                'state': LOCAL_DYNAMIC_OCCUPIED,
                                'timestamp': current_time,
                                'confidence': 0.5,  # Lower confidence for predictions (reduced from 0.6)
                                'obstacle_id': obstacle_id,
                                'predicted': True
                            }
        
        # Clear cells that were marked as dynamic obstacles but are no longer occupied
        # This allows update_from_global_ogm to update them to FREE on the next cycle
        cells_to_clear = []
        for (x, y), cell_data in list(self.local_map.items()):
            if cell_data.get('state') == LOCAL_DYNAMIC_OCCUPIED:
                # Check if this cell is still occupied by a current obstacle
                if (x, y) not in current_obstacle_cells:
                    # Cell is no longer occupied, clear it so it can be updated to FREE
                    cells_to_clear.append((x, y))
        
        # Clear the cells (they'll be updated to FREE by update_from_global_ogm on next update)
        for cell in cells_to_clear:
            del self.local_map[cell]
        
        # Clean up old obstacles
        self.obstacle_tracker.clear_old_obstacles(current_time, max_age=5.0)
    
    def get_cell_state(self, x, y):
        """
        Get cell state at local coordinates (global grid coordinates).
        
        Args:
            x: X coordinate in global grid
            y: Y coordinate in global grid
            
        Returns:
            Cell state (LOCAL_UNKNOWN, LOCAL_FREE, LOCAL_OCCUPIED, LOCAL_DYNAMIC_OCCUPIED)
        """
        if (x, y) not in self.local_map:
            return LOCAL_UNKNOWN
        
        cell_data = self.local_map[(x, y)]
        
        # Check confidence
        if cell_data['confidence'] < self.min_confidence:
            return LOCAL_UNKNOWN
        
        return cell_data['state']
    
    def is_traversable(self, x, y, allow_goals=True):
        """
        Check if cell is safe to traverse.
        
        Args:
            x: X coordinate in global grid
            y: Y coordinate in global grid
            allow_goals: If True, goal cells are traversable
            
        Returns:
            True if cell is traversable, False otherwise
        """
        if (x, y) not in self.local_map:
            return True  # Not in local map, assume traversable (will be checked by global OGM)
        
        cell_data = self.local_map[(x, y)]
        state = cell_data['state']
        confidence = cell_data.get('confidence', 1.0)
        is_predicted = cell_data.get('predicted', False)
        
        # Only block if confidence is high enough
        if confidence < self.min_confidence:
            return True  # Low confidence, don't block
        
        if state == LOCAL_FREE:
            return True
        elif state == LOCAL_DYNAMIC_OCCUPIED:
            # Only block if it's a current position (not predicted) or high confidence prediction
            if not is_predicted:
                return False  # Current obstacle position blocks path
            elif confidence > 0.8:
                return False  # High confidence prediction blocks path
            else:
                # Low confidence prediction, allow traversal (robot can try to go through)
                return True
        elif state == LOCAL_OCCUPIED:
            return False
        elif state == LOCAL_UNKNOWN:
            return True  # Unknown cells are traversable (will be checked by global OGM)
        
        return True
    
    def decay_observations(self, current_time, decay_rate=None):
        """
        Apply temporal decay to observations.
        
        Args:
            current_time: Current timestamp in milliseconds
            decay_rate: Optional decay rate override
        """
        if decay_rate is None:
            # Calculate decay rate from half-life
            # confidence = 0.5^(age / half_life)
            decay_rate = 0.5 ** (1.0 / (self.decay_half_life * 1000.0))  # Convert to per-millisecond
        
        cells_to_remove = []
        
        for (x, y), cell_data in self.local_map.items():
            age = current_time - cell_data['timestamp']
            
            # Apply exponential decay
            if age > 0:
                # confidence = initial_confidence * decay_rate^(age)
                # Using half-life formula: confidence = 0.5^(age / half_life_ms)
                half_life_ms = self.decay_half_life * 1000.0
                new_confidence = cell_data['confidence'] * (0.5 ** (age / half_life_ms))
                cell_data['confidence'] = new_confidence
            
            # Remove very old observations
            if cell_data['confidence'] < 0.1 or age > 5000:  # 5 seconds
                cells_to_remove.append((x, y))
        
        for cell in cells_to_remove:
            del self.local_map[cell]
    
    def get_local_map_window(self, robot_pos):
        """
        Get local map window around robot.
        
        Args:
            robot_pos: (x, y) robot position
            
        Returns:
            Dictionary of {(x, y): cell_data} for cells in local window
        """
        rx, ry = robot_pos
        window = {}
        
        for (x, y), cell_data in self.local_map.items():
            dist = math.sqrt((x - rx)**2 + (y - ry)**2)
            if dist <= self.radius:
                window[(x, y)] = cell_data.copy()
        
        return window
    
    def get_dynamic_obstacles_in_radius(self, robot_pos, radius=None):
        """
        Get dynamic obstacles within radius of robot.
        
        Args:
            robot_pos: (x, y) robot position
            radius: Search radius (defaults to local map radius)
            
        Returns:
            List of (obstacle_id, current_pos, predicted_pos) tuples
        """
        if radius is None:
            radius = self.radius
        
        return self.obstacle_tracker.get_dynamic_obstacles_in_radius(robot_pos, radius)
    
    def clear(self):
        """Clear local map and obstacle tracker."""
        self.local_map.clear()
        self.obstacle_tracker = DynamicObstacleTracker(max_history=10)
        debug_print("LocalMapper cleared")

