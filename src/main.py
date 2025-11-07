"""
Main entry point for the warehouse robot simulation.
Implements frontier-based exploration with iSAM localization.
"""

import pygame
import sys
import argparse
from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, GRID_SIZE,
    WHITE, BLACK, GREEN, RED,
    debug_log
)
from robot import Robot
from warehouse import Warehouse
from metrics import save_metrics_to_csv, get_next_run_number

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Warehouse Robot Simulation')
parser.add_argument('--algo', type=str, default='A*', 
                    choices=['A*', 'astar', 'A-star', 'A_star', 'dijkstra', 'Dijkstra'],
                    help='Pathfinding algorithm to use (default: A*)')
parser.add_argument('--map', type=str, default='map1',
                    choices=['map1', 'map2', 'map3', 'map4'],
                    help='Map configuration to use (default: map1)')
parser.add_argument('--obj', type=int, default=0,
                    help='Number of dynamic obstacles to spawn (default: 0, only spawns for map4 if not specified)')
parser.add_argument('--robot-speed', type=int, default=100,
                    help='Robot movement cooldown in milliseconds (default: 100ms, lower = faster)')
parser.add_argument('--obstacle-speed', type=int, default=350,
                    help='Dynamic obstacle movement cooldown in milliseconds (default: 350ms, lower = faster)')
args = parser.parse_args()

# Normalize algorithm name (robot expects: 'A*', 'DIJKSTRA')
algorithm = args.algo.replace('*', 'star').replace('-', '').replace('_', '').upper()
if algorithm == 'ASTAR':
    algorithm = 'A*'
elif algorithm == 'DIJKSTRA':
    algorithm = 'DIJKSTRA'

# Initialize pygame
pygame.init()

# Create the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(f"Warehouse Robot Simulation - {args.map.upper()} - {algorithm}")
clock = pygame.time.Clock()

# Initialize warehouse and robot
debug_log("=" * 50)
debug_log(f"INITIALIZING WAREHOUSE ROBOT SIMULATION WITH DFS EXPLORATION")
debug_log(f"Map: {args.map.upper()}")
debug_log(f"Pathfinding Algorithm: {algorithm}")
if args.obj > 0:
    debug_log(f"Dynamic Obstacles: {args.obj}")
debug_log(f"Robot Speed: {args.robot_speed}ms cooldown")
debug_log(f"Obstacle Speed: {args.obstacle_speed}ms cooldown")
debug_log("=" * 50)
warehouse = Warehouse(map_name=args.map, num_dynamic_obstacles=args.obj, obstacle_speed=args.obstacle_speed)
robot = Robot(1, 1, warehouse=warehouse, pathfinding_algorithm=algorithm, move_cooldown=args.robot_speed)

# Start autonomous mapping phase
debug_log("")
robot.start_mapping(warehouse)
debug_log("")

# Main game loop
running = True
while running:
    clock.tick(FPS)
    current_time = pygame.time.get_ticks()
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # Check mission status
    mission_complete = len(warehouse.goals) == 0 and robot.mapping_complete and not robot.is_mapping
    
    # Update robot rotation (only if mission not complete)
    if not mission_complete:
        robot.update_rotation(current_time)
    
    # Spawn dynamic obstacles when robot enters DELIVER_GOALS mode (if num_dynamic_obstacles > 0)
    if robot.exploration_mode == "DELIVER_GOALS" and warehouse.num_dynamic_obstacles > 0:
        if not warehouse.dynamic_obstacles_spawned:
            warehouse.spawn_dynamic_obstacles(num_obstacles=warehouse.num_dynamic_obstacles)
    
    # Update dynamic obstacles (only if spawned and mission not complete)
    if warehouse.dynamic_obstacles_spawned and not mission_complete:
        warehouse.update_dynamic_obstacles(current_time)
    
    # Update local map with dynamic obstacles and apply temporal decay (only if mission not complete)
    if not mission_complete:
        robot.update_local_map_with_dynamics(current_time)
    
    # Handle mapping phase or normal gameplay (only if mission not complete)
    if not mission_complete:
        if robot.is_mapping:
            # Check if path needs replanning due to dynamic obstacles
            robot.replan_if_needed(current_time)
            
            # Autonomous mapping phase - robot explores using DFS coverage
            robot.explore_next(current_time)
            
            # Periodic pose graph optimization (reduced frequency from 4s to 8s, only when needed)
            frame_count = pygame.time.get_ticks() // (1000 // 60)
            optimization_interval = 480  # Optimize every 480 frames (8 seconds at 60 FPS, reduced from 240 frames/4s)
            # Only optimize when needed (e.g., after loop closure detected)
            if frame_count % optimization_interval == 0 and robot.loop_closure_detected:
                # Convert warehouse obstacles to format for optimization
                obstacles = []
                for obs_x, obs_y in warehouse.obstacles:
                    obstacles.append((obs_x, obs_y))
                robot.isam.optimize_graph(obstacles, recent_nodes=10)
        else:
            # Normal gameplay - only allow manual control after mapping is complete
            if robot.mapping_complete:
                # Check if path needs replanning due to dynamic obstacles
                robot.replan_if_needed(current_time)
                
                # Handle continuous input
                keys = pygame.key.get_pressed()
                robot.handle_input(keys, warehouse, current_time)
                
                # Check for pickup (space bar)
                if keys[pygame.K_SPACE]:
                    if not robot.has_cargo:
                        robot.try_pickup(warehouse, current_time)
                
                # Check for drop (V key)
                if keys[pygame.K_v]:
                    if robot.has_cargo:
                        robot.try_drop(warehouse, current_time)
                
                # Auto-select closest goal if none selected
                if robot.current_goal is None and warehouse.goals:
                    robot.current_goal = warehouse.goals[0]
    
    # Clear screen
    screen.fill(WHITE)
    
    # Draw warehouse
    warehouse.draw(screen, robot)
    
    # Draw robot
    robot.draw(screen)
    
    # Draw current goal indicator
    if robot.current_goal:
        font_small = pygame.font.Font(None, 20)
        goal_x = int(robot.current_goal[0] * GRID_SIZE + GRID_SIZE // 2)
        goal_y = int(robot.current_goal[1] * GRID_SIZE - 10)
        pygame.draw.line(
            screen, RED,
            (int(robot.x * GRID_SIZE + GRID_SIZE // 2), int(robot.y * GRID_SIZE + GRID_SIZE // 2)),
            (goal_x, goal_y + 10),
            2
        )
    
    # Draw status information (player-facing only)
    font = pygame.font.Font(None, 36)
    
    # Check mission status
    mission_complete = len(warehouse.goals) == 0 and robot.mapping_complete and not robot.is_mapping
    
    if not mission_complete:
        # Show mapping progress during exploration
        if robot.is_mapping:
            explored_count = len(robot.visited) if hasattr(robot, 'visited') else 0
            total_free = len(warehouse.get_free_cells())
            progress = explored_count / total_free * 100 if total_free > 0 else 0
            
            progress_text = font.render(
                f"Mapping Progress: {progress:.1f}% ({explored_count}/{total_free} cells)",
                True, BLACK
            )
            screen.blit(progress_text, (10, 10))
            
            # Show discovered goals counter during exploration
            discovered_goals = len(robot.ogm.goals) if robot.ogm else 0
            goals_text = font.render(
                f"Goals Discovered: {discovered_goals}",
                True, BLACK
            )
            screen.blit(goals_text, (10, 50))
        else:
            # Show goals remaining and score after mapping is complete
            goals_remaining = len(warehouse.goals)
            if hasattr(robot, 'goals_to_deliver') and robot.goals_to_deliver:
                goals_remaining = len(robot.goals_to_deliver)
            
            status_text = font.render(
                f"Goals Remaining: {goals_remaining} | Score: {robot.score}",
                True, BLACK
            )
            screen.blit(status_text, (10, 10))
            
            # Show cargo status if applicable
            if robot.mapping_complete:
                cargo_text = font.render(
                    f"Cargo: {'Yes' if robot.has_cargo else 'No'}",
                    True, BLACK
                )
                screen.blit(cargo_text, (10, 50))
    
    # Show completion message and save metrics
    if mission_complete:
        victory_text = font.render("MISSION COMPLETE! All packages delivered!", True, GREEN)
        # Center the text
        text_rect = victory_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        screen.blit(victory_text, text_rect)
        
        # Save metrics if not already saved
        if not hasattr(robot, 'metrics_saved'):
            # Get metrics from robot
            metrics_data = robot.get_metrics_for_export()
            
            # Get run number
            run_number = get_next_run_number(algorithm, args.map)
            
            # Save metrics to CSV
            save_metrics_to_csv(algorithm, args.map, run_number, metrics_data)
            
            # Mark as saved to avoid saving multiple times
            robot.metrics_saved = True
    
    # Update display
    pygame.display.flip()

# Cleanup
pygame.quit()
sys.exit()
