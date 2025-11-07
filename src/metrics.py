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
    
    print(f"Metrics saved to: {filepath}")
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

