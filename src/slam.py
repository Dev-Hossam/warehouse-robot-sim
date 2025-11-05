"""
SLAM (Simultaneous Localization and Mapping) module for warehouse robot simulation.
Implements pose estimation, scan matching, and loop closure detection.
"""

import math

DEBUG = True  # Enable/disable debugging output

def debug_print(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[SLAM DEBUG] {message}")


class SLAMSystem:
    """
    SLAM system that coordinates localization and mapping.
    Tracks estimated pose with uncertainty and detects loop closures.
    """
    
    def __init__(self, initial_x=0, initial_y=0, initial_theta=0):
        """
        Initialize SLAM system.
        
        Args:
            initial_x: Initial x position
            initial_y: Initial y position
            initial_theta: Initial orientation in degrees
        """
        # Estimated pose (x, y, theta)
        self.estimated_x = initial_x
        self.estimated_y = initial_y
        self.estimated_theta = initial_theta
        
        # Pose uncertainty (covariance matrix elements)
        # For simplicity, we track position and orientation uncertainty separately
        self.position_uncertainty = 0.1  # Initial position uncertainty
        self.orientation_uncertainty = 5.0  # Initial orientation uncertainty in degrees
        
        # Store pose history for loop closure detection
        self.pose_history = []
        self.scan_history = []  # Store LIDAR scans at each pose
        self.max_history = 100  # Maximum number of poses to store
        
        # Loop closure detection parameters
        self.loop_closure_threshold = 0.3  # Distance threshold for loop closure (in cells)
        self.scan_match_threshold = 0.7  # Similarity threshold for scan matching
        
        debug_print(f"SLAM system initialized at ({initial_x}, {initial_y}), theta={initial_theta}°")
    
    def predict_pose(self, odometry_delta):
        """
        Predict new pose from odometry (motion model).
        
        Args:
            odometry_delta: Tuple (dx, dy, dtheta) from odometry
        
        Returns:
            tuple: (new_x, new_y, new_theta) - predicted pose
        """
        dx, dy, dtheta = odometry_delta
        
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
        
        # Increase uncertainty (odometry accumulates error)
        # Reduced uncertainty growth to slow down drift accumulation
        self.position_uncertainty += 0.03  # Reduced from 0.05
        self.orientation_uncertainty += 0.5  # Reduced from 1.0
        
        debug_print(f"Predicted pose: ({self.estimated_x:.3f}, {self.estimated_y:.3f}), "
                   f"theta={self.estimated_theta:.3f}°, uncertainty={self.position_uncertainty:.3f}")
        
        return (self.estimated_x, self.estimated_y, self.estimated_theta)
    
    def update_pose_from_sensor(self, scan_results, ogm):
        """
        Update pose estimate using sensor data (measurement model).
        This is a simplified version - in full SLAM, this would use EKF or particle filter.
        
        Args:
            scan_results: LIDAR scan results
            ogm: Occupancy Grid Map
        """
        # For now, we just reduce uncertainty slightly when we get sensor data
        # In a full implementation, this would correct the pose based on scan matching
        self.position_uncertainty = max(0.05, self.position_uncertainty * 0.95)
        self.orientation_uncertainty = max(1.0, self.orientation_uncertainty * 0.95)
        
        debug_print(f"Updated pose from sensor, uncertainty reduced to {self.position_uncertainty:.3f}")
    
    def get_estimated_pose(self):
        """
        Get current estimated pose.
        
        Returns:
            tuple: (x, y, theta) - estimated pose
        """
        return (self.estimated_x, self.estimated_y, self.estimated_theta)
    
    def get_uncertainty(self):
        """
        Get current pose uncertainty.
        
        Returns:
            tuple: (position_uncertainty, orientation_uncertainty)
        """
        return (self.position_uncertainty, self.orientation_uncertainty)
    
    def store_pose_and_scan(self, scan_results):
        """
        Store current pose and scan for loop closure detection.
        
        Args:
            scan_results: Current LIDAR scan results
        """
        pose = (self.estimated_x, self.estimated_y, self.estimated_theta)
        self.pose_history.append(pose)
        
        # Store a simplified version of the scan (just obstacles for comparison)
        # Use list() instead of deepcopy to avoid recursion issues with complex structures
        obstacles = scan_results.get('obstacles', [])
        scan_copy = {
            'obstacles': [tuple(obs) for obs in obstacles] if obstacles else [],  # Convert to tuples for safety
            'ray_hits': []  # Don't store ray_hits to avoid deepcopy issues
        }
        self.scan_history.append(scan_copy)
        
        # Limit history size
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
            self.scan_history.pop(0)
        
        debug_print(f"Stored pose and scan, history size: {len(self.pose_history)}")
    
    def compare_scans(self, scan1, scan2):
        """
        Compare two LIDAR scans and return similarity score.
        
        Args:
            scan1: First scan (dict with 'obstacles' and 'ray_hits')
            scan2: Second scan (dict with 'obstacles' and 'ray_hits')
        
        Returns:
            float: Similarity score (0.0-1.0), higher is more similar
        """
        # Compare obstacle sets
        obstacles1 = set(scan1.get('obstacles', []))
        obstacles2 = set(scan2.get('obstacles', []))
        
        if len(obstacles1) == 0 and len(obstacles2) == 0:
            return 1.0  # Both empty, consider similar
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(obstacles1 & obstacles2)
        union = len(obstacles1 | obstacles2)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        # Also compare ray hit patterns (simplified)
        ray_hits1 = scan1.get('ray_hits', [])
        ray_hits2 = scan2.get('ray_hits', [])
        
        if len(ray_hits1) > 0 and len(ray_hits2) > 0:
            # Compare distances (simplified feature matching)
            distances1 = sorted([hit.get('distance', 0) for hit in ray_hits1])
            distances2 = sorted([hit.get('distance', 0) for hit in ray_hits2])
            
            if len(distances1) == len(distances2):
                # Calculate normalized difference
                max_dist = max(max(distances1) if distances1 else 1, 
                              max(distances2) if distances2 else 1, 1)
                diff = sum(abs(d1 - d2) for d1, d2 in zip(distances1, distances2)) / (len(distances1) * max_dist)
                distance_similarity = 1.0 - min(1.0, diff)
                
                # Combine similarities
                similarity = 0.7 * similarity + 0.3 * distance_similarity
        
        return similarity
    
    def detect_loop_closure(self, current_scan):
        """
        Detect if robot has returned to a previously visited location (loop closure).
        
        Args:
            current_scan: Current LIDAR scan results
        
        Returns:
            tuple: (detected, matched_pose_index) or (False, None)
        """
        if len(self.pose_history) < 10:
            # Need enough history to detect loops
            return (False, None)
        
        # Compare current scan with older scans (skip recent 30% to avoid false positives)
        # Only check poses that are far enough back to be a real loop
        recent_start = max(0, len(self.pose_history) - int(self.max_history * 0.3))
        recent_end = len(self.pose_history) - 20  # Skip very recent poses (need at least 20 steps before loop)
        
        if recent_start >= recent_end:
            return (False, None)
        
        best_similarity = 0.0
        best_match_index = None
        
        # Compare with poses in the window (but skip if too close)
        for i in range(recent_start, recent_end):
            old_scan = self.scan_history[i]
            similarity = self.compare_scans(current_scan, old_scan)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = i
        
        # Check if similarity is high enough and poses are close enough
        # Use stricter threshold during exploration to avoid false positives
        if best_match_index is not None and best_similarity >= self.scan_match_threshold:
            matched_pose = self.pose_history[best_match_index]
            current_pose = (self.estimated_x, self.estimated_y, self.estimated_theta)
            
            # Calculate distance between poses
            dx = current_pose[0] - matched_pose[0]
            dy = current_pose[1] - matched_pose[1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Stricter distance check - must be close enough to be a real loop
            # Also require that we've moved significantly since the matched pose
            steps_since_match = len(self.pose_history) - best_match_index
            if distance <= self.loop_closure_threshold * 5 and steps_since_match >= 20:
                debug_print(f"Loop closure detected! Similarity: {best_similarity:.3f}, "
                           f"distance: {distance:.3f}, steps since match: {steps_since_match}, matched pose index: {best_match_index}")
                return (True, best_match_index)
        
        return (False, None)
    
    def correct_pose_loop_closure(self, matched_pose_index):
        """
        Correct pose estimate when loop closure is detected.
        
        Args:
            matched_pose_index: Index of matched pose in history
        """
        if matched_pose_index is None or matched_pose_index >= len(self.pose_history):
            return
        
        matched_pose = self.pose_history[matched_pose_index]
        current_pose = (self.estimated_x, self.estimated_y, self.estimated_theta)
        
        # Calculate pose correction (interpolate between current and matched pose)
        # This is a simplified correction - full SLAM would use more sophisticated methods
        correction_factor = 0.85  # 85% correction for better error correction
        
        dx = matched_pose[0] - current_pose[0]
        dy = matched_pose[1] - current_pose[1]
        dtheta = (matched_pose[2] - current_pose[2]) % 360
        if dtheta > 180:
            dtheta -= 360
        
        # Apply correction
        self.estimated_x += dx * correction_factor
        self.estimated_y += dy * correction_factor
        self.estimated_theta = (self.estimated_theta + dtheta * correction_factor) % 360
        
        # Reduce uncertainty after loop closure
        self.position_uncertainty = max(0.05, self.position_uncertainty * 0.5)
        self.orientation_uncertainty = max(1.0, self.orientation_uncertainty * 0.5)
        
        debug_print(f"Pose corrected: ({self.estimated_x:.3f}, {self.estimated_y:.3f}), "
                   f"theta={self.estimated_theta:.3f}°, uncertainty reduced")
    
    def set_pose(self, x, y, theta):
        """
        Set pose estimate directly (used for initialization or external correction).
        
        Args:
            x: X position
            y: Y position
            theta: Orientation in degrees
        """
        self.estimated_x = x
        self.estimated_y = y
        self.estimated_theta = theta % 360
        debug_print(f"SLAM pose set to: ({x}, {y}), theta={theta}°")
    
    def reset(self):
        """Reset SLAM system to initial state."""
        self.pose_history = []
        self.scan_history = []
        self.position_uncertainty = 0.1
        self.orientation_uncertainty = 5.0
        debug_print("SLAM system reset")

