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

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Warehouse Robot Simulation')
parser.add_argument('--algo', type=str, default='A*', 
                    choices=['A*', 'astar', 'A-star', 'A_star', 'dijkstra', 'Dijkstra', 
                            'rrt', 'RRT', 'prm', 'PRM'],
                    help='Pathfinding algorithm to use (default: A*)')
parser.add_argument('--map', type=str, default='map1',
                    choices=['map1', 'map2', 'map3', 'map4'],
                    help='Map configuration to use (default: map1)')
args = parser.parse_args()

# Normalize algorithm name (robot expects: 'A*', 'DIJKSTRA', 'RRT', 'PRM')
algorithm = args.algo.replace('*', 'star').replace('-', '').replace('_', '').upper()
if algorithm == 'ASTAR':
    algorithm = 'A*'
elif algorithm == 'DIJKSTRA':
    algorithm = 'DIJKSTRA'
elif algorithm == 'RRT':
    algorithm = 'RRT'
elif algorithm == 'PRM':
    algorithm = 'PRM'

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
debug_log("=" * 50)
warehouse = Warehouse(map_name=args.map)
robot = Robot(1, 1, warehouse=warehouse, pathfinding_algorithm=algorithm)

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
    
    # Update robot rotation
    robot.update_rotation(current_time)
    
    # Spawn dynamic obstacles for map4 when robot enters DELIVER_GOALS mode
    if robot.exploration_mode == "DELIVER_GOALS" and warehouse.map_name == 'map4':
        if not warehouse.dynamic_obstacles_spawned:
            warehouse.spawn_dynamic_obstacles(num_obstacles=8)
    
    # Update dynamic obstacles (only if spawned)
    if warehouse.dynamic_obstacles_spawned:
        warehouse.update_dynamic_obstacles(current_time)
    
    # Update local map with dynamic obstacles and apply temporal decay
    robot.update_local_map_with_dynamics(current_time)
    
    # Handle mapping phase or normal gameplay
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
    
    # Draw instructions and status
    font = pygame.font.Font(None, 24)
    if robot.is_mapping:
        text = font.render("DFS COVERAGE EXPLORATION IN PROGRESS...", True, RED)
        screen.blit(text, (10, 10))
        
        # Get exploration status
        explored_count = len(robot.visited) if hasattr(robot, 'visited') else 0
        total_free = len(warehouse.get_free_cells())
        stack_size = len(robot.stack) if hasattr(robot, 'stack') else 0
        
        mapping_status = font.render(
            f"Visited: {explored_count}/{total_free} | Stack: {stack_size} | Mode: {robot.exploration_mode}",
            True, BLACK
        )
        screen.blit(mapping_status, (10, 35))
        
        # iSAM pose information
        estimated_pose = robot.isam.get_estimated_pose()
        uncertainty = robot.isam.get_uncertainty()
        robot_pos = font.render(
            f"Actual: ({int(robot.x)}, {int(robot.y)}) | Est: ({estimated_pose[0]:.1f}, {estimated_pose[1]:.1f}) | Angle: {int(robot.rotation_angle)}°",
            True, BLACK
        )
        screen.blit(robot_pos, (10, 60))
        
        isam_status = font.render(
            f"Uncertainty: {uncertainty[0]:.3f} | Loop Closure: {'DETECTED' if robot.loop_closure_detected else 'None'}",
            True, BLACK
        )
        screen.blit(isam_status, (10, 85))
        
        # Show mapping progress
        progress = explored_count / total_free * 100 if total_free > 0 else 0
        progress_text = font.render(
            f"Mapping Progress: {progress:.1f}% ({explored_count}/{total_free} cells)",
            True, BLACK
        )
        screen.blit(progress_text, (10, 110))
        
        # Show DFS status, return status, or delivery status
        if robot.exploration_mode == "RETURN_TO_START":
            if hasattr(robot, 'return_path') and robot.return_path:
                remaining = len(robot.return_path) - robot.return_path_index
                return_text = font.render(
                    f"RETURNING TO START: {remaining} steps remaining",
                    True, GREEN
                )
                screen.blit(return_text, (10, 135))
        elif robot.exploration_mode == "DELIVER_GOALS":
            if hasattr(robot, 'goals_to_deliver'):
                goals_remaining = len(robot.goals_to_deliver)
                goals_total = len(warehouse.goals) + goals_remaining if warehouse.goals else goals_remaining
                delivery_text = font.render(
                    f"DELIVERING GOALS: {goals_remaining} remaining | Mode: {robot.delivery_mode} | Score: {robot.score}",
                    True, GREEN
                )
                screen.blit(delivery_text, (10, 135))
                if hasattr(robot, 'delivery_path') and robot.delivery_path:
                    remaining = len(robot.delivery_path) - robot.delivery_path_index
                    path_text = font.render(
                        f"Path steps remaining: {remaining}",
                        True, BLACK
                    )
                    screen.blit(path_text, (10, 160))
        elif hasattr(robot, 'stack') and robot.stack:
            dfs_text = font.render(
                f"DFS Stack: {len(robot.stack)} cells | Backtracking when needed",
                True, BLACK
            )
            screen.blit(dfs_text, (10, 135))
    else:
        text = font.render("Arrow Keys: Move | Space: Pickup | V: Drop | ESC: Quit", True, BLACK)
        screen.blit(text, (10, 10))
        
        # iSAM pose information
        estimated_pose = robot.isam.get_estimated_pose()
        uncertainty = robot.isam.get_uncertainty()
        robot_pos = font.render(
            f"Actual: ({int(robot.x)}, {int(robot.y)}) | Est: ({estimated_pose[0]:.1f}, {estimated_pose[1]:.1f}) | Cargo: {'Yes' if robot.has_cargo else 'No'} | Angle: {int(robot.rotation_angle)}°",
            True, BLACK
        )
        screen.blit(robot_pos, (10, 35))
        
        status = font.render(
            f"Goals Remaining: {len(warehouse.goals)} | Score: {robot.score} | Uncertainty: {uncertainty[0]:.3f}",
            True, BLACK
        )
        screen.blit(status, (10, 60))
        
        if robot.loop_closure_detected:
            loop_text = font.render("LOOP CLOSURE DETECTED!", True, (255, 0, 255))
            screen.blit(loop_text, (10, 85))
        
        if robot.mapping_complete:
            mapping_done = font.render("MAPPING COMPLETE - All obstacles and goals identified!", True, GREEN)
            screen.blit(mapping_done, (10, 110))
    
    # Legend
    legend_y = SCREEN_HEIGHT - 100
    legend_font = pygame.font.Font(None, 20)
    legend1 = legend_font.render("Blue circle = Actual robot position", True, BLACK)
    screen.blit(legend1, (10, legend_y))
    legend2 = legend_font.render("Green circle = Estimated pose (iSAM)", True, BLACK)
    screen.blit(legend2, (10, legend_y + 20))
    legend3 = legend_font.render("Orange line = Pose error", True, BLACK)
    screen.blit(legend3, (10, legend_y + 40))
    legend4 = legend_font.render("Magenta circle = Loop closure detected", True, BLACK)
    screen.blit(legend4, (10, legend_y + 60))
    
    # Show completion message
    if len(warehouse.goals) == 0:
        victory_text = font.render("MISSION COMPLETE! All packages delivered!", True, GREEN)
        screen.blit(victory_text, (300, 300))
    
    # Update display
    pygame.display.flip()

# Cleanup
pygame.quit()
sys.exit()
