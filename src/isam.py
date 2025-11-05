"""
iSAM (Incremental Smoothing and Mapping) module for warehouse robot simulation.
Implements pose graph optimization using networkx and scipy.optimize.least_squares.
"""

import math
import logging
import networkx as nx
import numpy as np
from scipy.optimize import least_squares

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEBUG = True

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[ISAM DEBUG] {message}")


class ISAM:
    """
    Incremental Smoothing and Mapping (iSAM) system using pose graph optimization.
    Uses networkx for pose graph and scipy.optimize.least_squares for optimization.
    """
    
    def __init__(self, initial_x=0, initial_y=0, initial_theta=0):
        """
        Initialize iSAM system.
        
        Args:
            initial_x: Initial x position
            initial_y: Initial y position
            initial_theta: Initial orientation in degrees
        """
        # Pose graph using networkx
        self.pose_graph = nx.Graph()
        
        # Current pose tracking
        self.estimated_x = initial_x
        self.estimated_y = initial_y
        self.estimated_theta = initial_theta
        
        # Node tracking
        self.node_id = 0
        self.previous_node = None
        
        # Position and angle tracking for loop closure
        self.previous_positions = {}
        self.previous_angles = {}
        
        # Loop closure tracking
        self.last_loop_closure_pos = np.array([initial_x, initial_y])
        self.last_loop_closure_angle = initial_theta
        
        # Optimization state
        self.optimization_in_progress = False
        
        # Robot movement parameters
        self.robot_speed = 1.0  # Grid cells per step
        
        debug_print(f"iSAM initialized at ({initial_x}, {initial_y}), theta={initial_theta}°")
    
    def add_node(self, pos, angle):
        """
        Add a node to the pose graph with position and angle tracking.
        
        Args:
            pos: Position as numpy array [x, y]
            angle: Orientation angle in degrees
        """
        node_id = self.node_id
        self.pose_graph.add_node(node_id, pos=pos.copy(), angle=angle)
        self.previous_positions[node_id] = pos.copy()
        self.previous_angles[node_id] = angle
        
        # Add edge from previous node if exists
        if self.previous_node is not None:
            self.pose_graph.add_edge(self.previous_node, node_id)
        
        self.previous_node = node_id
        self.node_id += 1
        
        debug_print(f"Added node {node_id-1} at ({pos[0]:.2f}, {pos[1]:.2f}), angle={angle:.2f}°")
    
    def detect_loop_closure(self, current_pos, current_angle, threshold=50, min_distance_since_last=150, angle_threshold=30):
        """
        Detect loop closure by checking proximity and orientation alignment with previous nodes.
        
        Args:
            current_pos: Current position as numpy array [x, y]
            current_angle: Current orientation angle in degrees
            threshold: Distance threshold for loop closure (in pixels/grid units)
            min_distance_since_last: Minimum distance since last loop closure
            angle_threshold: Angle difference threshold in degrees
            
        Returns:
            int or None: Node ID of matched node, or None if no loop closure
        """
        distance_since_last = np.linalg.norm(current_pos - self.last_loop_closure_pos)
        angle_diff_since_last = abs(current_angle - self.last_loop_closure_angle)
        angle_diff_since_last = min(angle_diff_since_last, 360 - angle_diff_since_last)
        
        if distance_since_last < min_distance_since_last and angle_diff_since_last < angle_threshold:
            return None
        
        # Limit search to 30 most recent nodes instead of all nodes (O(n) → O(30))
        recent_nodes = list(self.pose_graph.nodes)[-30:] if len(self.pose_graph.nodes) > 30 else list(self.pose_graph.nodes)
        
        for n in recent_nodes:
            if n == self.previous_node:
                continue
            
            pos = self.pose_graph.nodes[n]['pos']
            angle = self.pose_graph.nodes[n].get('angle', 0)
            distance = np.linalg.norm(current_pos - pos)
            angle_diff = abs(current_angle - angle)
            angle_diff = min(angle_diff, 360 - angle_diff)
            
            if distance < threshold and angle_diff < angle_threshold:
                logging.info(f"Loop closure detected between node {self.node_id-1} and node {n}.")
                self.last_loop_closure_pos = current_pos.copy()
                self.last_loop_closure_angle = current_angle
                return n
        
        return None
    
    def optimize_graph(self, obstacles, recent_nodes=10):
        """
        Perform pose graph optimization on the last 'recent_nodes' nodes.
        
        Args:
            obstacles: List of obstacle rectangles (for collision avoidance)
            recent_nodes: Number of recent nodes to optimize
        """
        if self.optimization_in_progress:
            return
        
        self.optimization_in_progress = True
        
        nodes = list(self.pose_graph.nodes)
        if len(nodes) < 2:
            self.optimization_in_progress = False
            return
        
        subset_nodes = nodes[-recent_nodes:] if len(nodes) > recent_nodes else nodes
        subset_graph = self.pose_graph.subgraph(subset_nodes)
        subset_nodes = list(subset_graph.nodes)
        
        if len(subset_nodes) < 2:
            self.optimization_in_progress = False
            return
        
        subset_poses = np.array([self.pose_graph.nodes[n]['pos'] for n in subset_nodes])
        subset_angles = np.array([self.pose_graph.nodes[n]['angle'] for n in subset_nodes])
        
        def error_function(x):
            """Error function for optimization."""
            errors = []
            updated_poses = x[:len(subset_nodes)*2].reshape((-1, 2))
            updated_angles = x[len(subset_nodes)*2:].reshape((-1, 1)).flatten()
            
            # Edge constraints
            for edge in subset_graph.edges(data=True):
                n1, n2, attrs = edge
                if n1 in subset_nodes and n2 in subset_nodes:
                    idx1 = subset_nodes.index(n1)
                    idx2 = subset_nodes.index(n2)
                    pos1 = updated_poses[idx1]
                    pos2 = updated_poses[idx2]
                    angle1 = updated_angles[idx1]
                    angle2 = updated_angles[idx2]
                    
                    # Desired distance constraint
                    desired_distance = self.robot_speed * 1
                    actual_distance = np.linalg.norm(pos2 - pos1)
                    errors.append(actual_distance - desired_distance)
                    
                    # Angle difference constraint
                    desired_angle_diff = (angle2 - angle1) % 360
                    desired_angle_diff = min(desired_angle_diff, 360 - desired_angle_diff)
                    errors.append(desired_angle_diff)
            
            # Obstacle avoidance constraints
            for pos in updated_poses:
                min_distance = float('inf')
                for obstacle in obstacles:
                    if hasattr(obstacle, 'centerx'):
                        # pygame.Rect
                        obstacle_center = np.array([obstacle.centerx, obstacle.centery])
                        distance = np.linalg.norm(pos - obstacle_center)
                        obstacle_min_distance = math.sqrt((obstacle.width / 2) ** 2 + (obstacle.height / 2) ** 2) + 5
                        if distance < obstacle_min_distance:
                            min_distance = min(min_distance, obstacle_min_distance - distance)
                    elif isinstance(obstacle, tuple):
                        # (x, y) tuple
                        obstacle_center = np.array([obstacle[0], obstacle[1]])
                        distance = np.linalg.norm(pos - obstacle_center)
                        obstacle_min_distance = 5  # Grid cell radius
                        if distance < obstacle_min_distance:
                            min_distance = min(min_distance, obstacle_min_distance - distance)
                
                if min_distance < float('inf'):
                    errors.append(min_distance * 10)
            
            return errors
        
        # Store previous positions before optimization
        for n in subset_nodes:
            self.previous_positions[n] = self.pose_graph.nodes[n]['pos'].copy()
            self.previous_angles[n] = self.pose_graph.nodes[n]['angle']
        
        # Initial guess
        x0 = np.hstack((subset_poses.flatten(), subset_angles))
        
        try:
            res = least_squares(error_function, x0, verbose=0, method='trf')
            optimized_poses = res.x[:len(subset_nodes)*2].reshape((-1, 2))
            optimized_angles = res.x[len(subset_nodes)*2:].reshape((-1, 1)).flatten()
            
            # Update graph with optimized poses
            for idx, n in enumerate(subset_nodes):
                self.previous_positions[n] = self.pose_graph.nodes[n]['pos'].copy()
                self.previous_angles[n] = self.pose_graph.nodes[n]['angle']
                self.pose_graph.nodes[n]['pos'] = optimized_poses[idx]
                self.pose_graph.nodes[n]['angle'] = optimized_angles[idx] % 360
                self.pose_graph.nodes[n]['optimized'] = True
            
            logging.info("Pose graph optimization completed.")
        except Exception as e:
            logging.error(f"Optimization failed: {e}")
        
        self.optimization_in_progress = False
    
    def get_estimated_pose(self):
        """
        Get current estimated pose.
        
        Returns:
            tuple: (x, y, theta) - estimated pose
        """
        return (self.estimated_x, self.estimated_y, self.estimated_theta)
    
    def set_pose(self, x, y, theta):
        """
        Set pose estimate directly.
        
        Args:
            x: X position
            y: Y position
            theta: Orientation in degrees
        """
        self.estimated_x = x
        self.estimated_y = y
        self.estimated_theta = theta % 360
        debug_print(f"iSAM pose set to: ({x}, {y}), theta={theta}°")
    
    def update_pose(self, dx, dy, dtheta):
        """
        Update pose estimate from odometry.
        
        Args:
            dx: Change in x
            dy: Change in y
            dtheta: Change in orientation in degrees
        """
        # Convert current theta to radians
        theta_rad = math.radians(self.estimated_theta)
        
        # Rotate movement vector by current orientation
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        
        # Apply rotation to movement vector
        rotated_dx = dx * cos_theta - dy * sin_theta
        rotated_dy = dx * sin_theta + dy * cos_theta
        
        # Update estimated pose
        self.estimated_x += rotated_dx
        self.estimated_y += rotated_dy
        self.estimated_theta = (self.estimated_theta + dtheta) % 360
        
        debug_print(f"Updated pose: ({self.estimated_x:.3f}, {self.estimated_y:.3f}), theta={self.estimated_theta:.3f}°")
    
    def get_uncertainty(self):
        """
        Get current pose uncertainty (simplified - returns fixed values).
        
        Returns:
            tuple: (position_uncertainty, orientation_uncertainty)
        """
        # For iSAM, uncertainty is reduced through optimization
        # Return small fixed values for visualization
        return (0.1, 5.0)
    
    def reset(self):
        """Reset iSAM system to initial state."""
        self.pose_graph = nx.Graph()
        self.node_id = 0
        self.previous_node = None
        self.previous_positions = {}
        self.previous_angles = {}
        self.optimization_in_progress = False
        debug_print("iSAM system reset")

