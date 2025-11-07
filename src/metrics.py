"""
Metrics tracking and export module for warehouse robot simulation.
Handles saving evaluation metrics to text files with formatted tables.
"""

import os
from datetime import datetime


def ensure_metrics_directory():
    """Ensure the metrics directory exists."""
    metrics_dir = "metrics"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    return metrics_dir


def ensure_goal_tracking_directory():
    """Ensure the goal tracking sub-directory exists."""
    metrics_dir = ensure_metrics_directory()
    goal_tracking_dir = os.path.join(metrics_dir, "goal_tracking")
    if not os.path.exists(goal_tracking_dir):
        os.makedirs(goal_tracking_dir)
    return goal_tracking_dir


def save_metrics_to_csv(algorithm, map_name, run_number, metrics_data):
    """
    Save metrics to a text file with formatted table.
    
    Args:
        algorithm: Algorithm name (e.g., 'A*', 'Dijkstra')
        map_name: Map name (e.g., 'map1', 'map2')
        run_number: Run number (e.g., 1, 2, 3)
        metrics_data: Dictionary containing metrics
    """
    metrics_dir = ensure_metrics_directory()
    
    # Format algorithm name for filename (replace * with Astar, replace spaces with _)
    algo_name = algorithm.replace('*', 'Astar').replace(' ', '_')
    
    # Create filename: Algorithm_Map{map_name}_n{run_number}.txt
    # Example: Astar_Map1_n1.txt or Dijkstra_Map1_n1.txt
    map_num = map_name.replace('map', '')
    filename = f"{algo_name}_Map{map_num}_n{run_number}.txt"
    filepath = os.path.join(metrics_dir, filename)
    
    # Extract metrics
    total_goals = metrics_data.get('total_goals_delivered', 0)
    total_time = metrics_data.get('total_time', 0)
    total_obstacles = metrics_data.get('total_obstacles', 0)
    total_dynamic_obstacles = metrics_data.get('total_dynamic_obstacles', 0)
    
    # Speed settings
    robot_speed = metrics_data.get('robot_speed', 100)
    obstacle_speed = metrics_data.get('obstacle_speed', 350)
    
    # Actual traversal metrics
    total_cells_traversed = metrics_data.get('total_cells_actually_traversed', 0)
    actual_avg_path_length = metrics_data.get('actual_avg_path_length_per_goal', 0)
    
    # Planned metrics
    planned_avg_path_length = metrics_data.get('planned_avg_path_length_per_goal', 0)
    
    # Replanning metrics
    total_replans = metrics_data.get('total_replans', 0)
    avg_replans_per_goal = metrics_data.get('avg_replans_per_goal', 0)
    
    # Create formatted table with clear sections
    table_lines = [
        "=" * 80,
        f"METRICS REPORT - {algorithm} on {map_name.upper()} - Run #{run_number}",
        "=" * 80,
        "",
        f"{'Metric':<55} {'Value':>25}",
        "=" * 80,
        "",
        "MISSION SUMMARY",
        "-" * 80,
        f"{'Total Goals Delivered':<55} {total_goals:>25}",
        f"{'Total Time Taken (seconds)':<55} {total_time:>25.2f}",
        f"{'Total Static Obstacles':<55} {total_obstacles:>25}",
        f"{'Total Dynamic Obstacles':<55} {total_dynamic_obstacles:>25}",
        "",
        "SPEED SETTINGS",
        "-" * 80,
        f"{'Robot Speed (ms cooldown)':<55} {robot_speed:>25}",
        f"{'Obstacle Speed (ms cooldown)':<55} {obstacle_speed:>25}",
        "",
        "ACTUAL TRAVERSAL (What Robot Actually Did)",
        "-" * 80,
        f"{'Total Cells Actually Traversed':<55} {total_cells_traversed:>25}",
        f"{'Average Path Length per Goal (actual)':<55} {actual_avg_path_length:>25.2f}",
        "",
        "PLANNED vs ACTUAL COMPARISON",
        "-" * 80,
        f"{'Planned Avg Path Length per Goal':<55} {planned_avg_path_length:>25.2f}",
        f"{'Actual Avg Path Length per Goal':<55} {actual_avg_path_length:>25.2f}",
        f"{'Difference (actual - planned)':<55} {actual_avg_path_length - planned_avg_path_length:>25.2f}",
        "",
        "REPLANNING METRICS",
        "-" * 80,
        f"{'Total Number of Replans':<55} {total_replans:>25}",
        f"{'Average Replans per Goal':<55} {avg_replans_per_goal:>25.2f}",
        "=" * 80,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]
    
    # Write to text file
    with open(filepath, 'w', encoding='utf-8') as txtfile:
        txtfile.write('\n'.join(table_lines))
    
    # Save goal tracking data to separate file
    goals_data = metrics_data.get('goals_data', [])
    if goals_data:
        start_time = metrics_data.get('start_time', None)
        save_goal_tracking_data(algorithm, map_name, run_number, goals_data, start_time)
    
    print(f"Metrics saved to: {filepath}")
    return filepath


def save_goal_tracking_data(algorithm, map_name, run_number, goals_data, start_time=None):
    """
    Save goal tracking data to a separate file in the goal_tracking sub-folder.
    
    Args:
        algorithm: Algorithm name (e.g., 'A*', 'Dijkstra')
        map_name: Map name (e.g., 'map1', 'map2')
        run_number: Run number (e.g., 1, 2, 3)
        goals_data: List of goal statistics dictionaries
        start_time: Mission start timestamp (optional, for relative time calculation)
    """
    goal_tracking_dir = ensure_goal_tracking_directory()
    
    # Format algorithm name for filename (replace * with Astar, replace spaces with _)
    algo_name = algorithm.replace('*', 'Astar').replace(' ', '_')
    
    # Create filename: Algorithm_Map{map_name}_n{run_number}_goals.txt
    map_num = map_name.replace('map', '')
    filename = f"{algo_name}_Map{map_num}_n{run_number}_goals.txt"
    filepath = os.path.join(goal_tracking_dir, filename)
    
    # Create formatted table
    table_lines = [
        "=" * 100,
        f"GOAL TRACKING REPORT - {algorithm} on {map_name.upper()} - Run #{run_number}",
        "=" * 100,
        "",
        f"{'Goal #':<8} {'Goal Location':<18} {'Pickup Time (s)':<18} {'Discharge Location':<20} {'Discharge Time (s)':<20}",
        "=" * 100,
        ""
    ]
    
    # Determine reference time for relative calculations
    # Always use start_time as the reference (when robot starts moving/delivery phase)
    # This ensures pickup times are relative to mission start, not first pickup
    reference_time = start_time
    
    # If start_time is None, we need to find the earliest event to use as reference
    # This should rarely happen, but we handle it gracefully
    if reference_time is None and goals_data:
        # Find the earliest timestamp across all goals (pickup or discharge)
        earliest_time = None
        for goal_stat in goals_data:
            pickup_ts = goal_stat.get('pickup_timestamp', 0)
            discharge_ts = goal_stat.get('discharge_timestamp', 0)
            for ts in [pickup_ts, discharge_ts]:
                if ts > 0:
                    if earliest_time is None or ts < earliest_time:
                        earliest_time = ts
        reference_time = earliest_time
        if reference_time is None:
            print("WARNING: No start_time provided and no valid timestamps found in goals_data")
    
    # Add goal data rows
    for idx, goal_stat in enumerate(goals_data, 1):
        goal_location = goal_stat.get('goal_location', goal_stat.get('goal', (0, 0)))
        pickup_timestamp = goal_stat.get('pickup_timestamp', 0)
        discharge_location = goal_stat.get('discharge_location', (0, 0))
        discharge_timestamp = goal_stat.get('discharge_timestamp', 0)
        
        # Calculate relative times (elapsed seconds from mission start)
        if reference_time is not None:
            pickup_time_relative = pickup_timestamp - reference_time if pickup_timestamp > 0 else 0.0
            discharge_time_relative = discharge_timestamp - reference_time if discharge_timestamp > 0 else 0.0
        else:
            # Fallback: if no reference time available, show 0.0
            pickup_time_relative = 0.0
            discharge_time_relative = 0.0
        
        goal_loc_str = f"({goal_location[0]},{goal_location[1]})"
        discharge_loc_str = f"({discharge_location[0]},{discharge_location[1]})"
        
        table_lines.append(
            f"{idx:<8} {goal_loc_str:<18} {pickup_time_relative:<18.2f} {discharge_loc_str:<20} {discharge_time_relative:<20.2f}"
        )
    
    table_lines.extend([
        "=" * 100,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ])
    
    # Write to text file
    with open(filepath, 'w', encoding='utf-8') as txtfile:
        txtfile.write('\n'.join(table_lines))
    
    print(f"Goal tracking data saved to: {filepath}")
    return filepath


def get_next_run_number(algorithm, map_name):
    """
    Get the next run number for a given algorithm and map combination.
    
    Args:
        algorithm: Algorithm name
        map_name: Map name
        
    Returns:
        int: Next run number
    """
    metrics_dir = ensure_metrics_directory()
    # Format algorithm name same way as in save_metrics_to_csv
    algo_name = algorithm.replace('*', 'Astar').replace(' ', '_')
    map_num = map_name.replace('map', '')
    
    # Find existing files with this pattern
    pattern = f"{algo_name}_Map{map_num}_n"
    run_numbers = []
    
    if os.path.exists(metrics_dir):
        for filename in os.listdir(metrics_dir):
            if filename.startswith(pattern) and filename.endswith('.txt'):
                # Extract run number from filename
                try:
                    # Format: Algorithm_Map1_n1.txt
                    parts = filename.replace('.txt', '').split('_n')
                    if len(parts) == 2:
                        run_num = int(parts[1])
                        run_numbers.append(run_num)
                except (ValueError, IndexError):
                    continue
    
    if run_numbers:
        return max(run_numbers) + 1
    else:
        return 1

