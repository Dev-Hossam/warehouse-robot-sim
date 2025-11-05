import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60
GRID_SIZE = 40

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
BLUE = (0, 100, 255)
GREEN = (0, 200, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
LOADING_DOCK = (255, 165, 0)  # Orange
DISCHARGE_DOCK = (200, 0, 200)  # Purple
GOAL_COLOR = (0, 255, 0)  # Bright green

# Warehouse grid dimensions
WAREHOUSE_WIDTH = SCREEN_WIDTH // GRID_SIZE
WAREHOUSE_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 1
        self.current_goal = None
        self.has_cargo = False
        self.last_move_time = 0
        self.move_cooldown = 200  # milliseconds between moves
        self.last_action_time = 0
        self.action_cooldown = 300  # milliseconds between actions
        self.score = 0
    
    def handle_input(self, keys, warehouse, current_time):
        # Only allow movement if enough time has passed
        if current_time - self.last_move_time < self.move_cooldown:
            return
        
        new_x, new_y = self.x, self.y
        moved = False
        
        if keys[pygame.K_UP] and self.y > 0:
            new_y -= self.speed
            moved = True
        elif keys[pygame.K_DOWN] and self.y < WAREHOUSE_HEIGHT - 1:
            new_y += self.speed
            moved = True
        elif keys[pygame.K_LEFT] and self.x > 0:
            new_x -= self.speed
            moved = True
        elif keys[pygame.K_RIGHT] and self.x < WAREHOUSE_WIDTH - 1:
            new_x += self.speed
            moved = True
        
        # Only move if the new position is not blocked
        if moved and not warehouse.is_blocked(new_x, new_y):
            self.x, self.y = new_x, new_y
            self.last_move_time = current_time
    
    def try_pickup(self, warehouse, current_time):
        if current_time - self.last_action_time < self.action_cooldown:
            return False
        
        if self.has_cargo or not self.current_goal:
            return False
        
        goal_x, goal_y = self.current_goal
        if abs(self.x - goal_x) < 1.5 and abs(self.y - goal_y) < 1.5:
            self.has_cargo = True
            warehouse.goals.remove(self.current_goal)
            self.last_action_time = current_time
            return True
        return False
    
    def try_drop(self, warehouse, current_time):
        if current_time - self.last_action_time < self.action_cooldown:
            return False
        
        if not self.has_cargo:
            return False
        
        dock_x, dock_y = warehouse.discharge_dock
        if abs(self.x - dock_x) < 1.5 and abs(self.y - dock_y) < 1.5:
            self.has_cargo = False
            self.score += 1  # Increment score for each delivery
            self.last_action_time = current_time
            # Update current goal to the next one
            if warehouse.goals:
                self.current_goal = warehouse.goals[0]
            else:
                self.current_goal = None
            return True
        return False
    
    def draw(self, surface):
        x_pixel = self.x * GRID_SIZE
        y_pixel = self.y * GRID_SIZE
        pygame.draw.circle(
            surface,
            BLUE,
            (x_pixel + GRID_SIZE // 2, y_pixel + GRID_SIZE // 2),
            GRID_SIZE // 3
        )
        
        # Draw cargo indicator
        if self.has_cargo:
            pygame.draw.circle(
                surface,
                RED,
                (x_pixel + GRID_SIZE // 2, y_pixel + GRID_SIZE // 2),
                GRID_SIZE // 5
            )

class Warehouse:
    def __init__(self):
        self.obstacles = set()
        self.goals = []
        self.loading_dock = None
        self.discharge_dock = None
        self.create_maze_layout()
        self.create_docks_and_goals()
    
    def create_maze_layout(self):
        # Create walls around the perimeter
        for x in range(WAREHOUSE_WIDTH):
            self.obstacles.add((x, 0))
            self.obstacles.add((x, WAREHOUSE_HEIGHT - 1))
        for y in range(WAREHOUSE_HEIGHT):
            self.obstacles.add((0, y))
            self.obstacles.add((WAREHOUSE_WIDTH - 1, y))
        
        # Create internal maze structure with wider paths
        # Horizontal corridors (skip every 5th row for wider paths)
        for y in [4, 8, 12]:
            if y < WAREHOUSE_HEIGHT:
                for x in range(2, WAREHOUSE_WIDTH - 2):
                    if x % 8 not in [0, 1]:  # Wider gaps for vertical passages
                        self.obstacles.add((x, y))
        
        # Vertical corridors (wider spacing)
        for x in [7, 15]:
            if x < WAREHOUSE_WIDTH:
                for y in range(2, WAREHOUSE_HEIGHT - 2):
                    if y % 6 not in [0, 1]:  # Wider gaps for horizontal passages
                        self.obstacles.add((x, y))
        
        # Add strategic obstacles
        maze_obstacles = [
            (4, 6), (10, 6), (18, 6),
            (4, 10), (12, 10), (20, 10),
            (6, 14), (14, 14), (22, 14),
        ]
        for x, y in maze_obstacles:
            if x < WAREHOUSE_WIDTH and y < WAREHOUSE_HEIGHT:
                self.obstacles.add((x, y))
    
    def create_docks_and_goals(self):
        # Loading dock at top left area
        self.loading_dock = (2, 2)
        
        # Discharge dock at bottom right area
        self.discharge_dock = (WAREHOUSE_WIDTH - 3, WAREHOUSE_HEIGHT - 3)
        
        # Create goal points (packages to collect)
        self.goals = [
            (5, 5), (9, 3), (13, 6),
            (17, 5), (19, 9), (11, 11),
            (7, 13), (21, 13)
        ]
    
    def is_blocked(self, x, y):
        return (int(x), int(y)) in self.obstacles
    
    def draw(self, surface, robot):
        # Draw grid
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(surface, GRAY, (x, 0), (x, SCREEN_HEIGHT), 1)
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(surface, GRAY, (0, y), (SCREEN_WIDTH, y), 1)
        
        # Draw loading dock
        if self.loading_dock:
            x = self.loading_dock[0] * GRID_SIZE
            y = self.loading_dock[1] * GRID_SIZE
            pygame.draw.rect(surface, LOADING_DOCK, (x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2))
        
        # Draw discharge dock (highlight if robot has cargo)
        if self.discharge_dock:
            x = self.discharge_dock[0] * GRID_SIZE
            y = self.discharge_dock[1] * GRID_SIZE
            color = (255, 100, 100) if robot.has_cargo else DISCHARGE_DOCK
            pygame.draw.rect(surface, color, (x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2))
            if robot.has_cargo:
                pygame.draw.rect(surface, RED, (x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2), 3)
        
        # Draw goal points (packages) with priority numbers
        font_small = pygame.font.Font(None, 18)
        for idx, (col, row) in enumerate(self.goals):
            x = col * GRID_SIZE
            y = row * GRID_SIZE
            pygame.draw.rect(surface, GOAL_COLOR, (x + 4, y + 4, GRID_SIZE - 8, GRID_SIZE - 8))
            # Draw priority number
            priority_text = font_small.render(str(idx + 1), True, BLACK)
            text_rect = priority_text.get_rect(center=(x + GRID_SIZE // 2, y + GRID_SIZE // 2))
            surface.blit(priority_text, text_rect)
        
        # Draw obstacles
        for col, row in self.obstacles:
            x = col * GRID_SIZE
            y = row * GRID_SIZE
            pygame.draw.rect(surface, YELLOW, (x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2))
            pygame.draw.rect(surface, DARK_GRAY, (x + 1, y + 1, GRID_SIZE - 2, GRID_SIZE - 2), 1)

# Create the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Warehouse Robot Simulation")
clock = pygame.time.Clock()

# Initialize warehouse and robot
warehouse = Warehouse()
robot = Robot(1, 1)

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
        pygame.draw.line(screen, RED, (int(robot.x * GRID_SIZE + GRID_SIZE // 2), int(robot.y * GRID_SIZE + GRID_SIZE // 2)), (goal_x, goal_y + 10), 2)
    
    # Draw instructions
    font = pygame.font.Font(None, 24)
    text = font.render("Arrow Keys: Move | Space: Pick/Drop | ESC: Quit", True, BLACK)
    screen.blit(text, (10, 10))
    robot_pos = font.render(f"Robot: ({int(robot.x)}, {int(robot.y)}) | Cargo: {'Yes' if robot.has_cargo else 'No'}", True, BLACK)
    screen.blit(robot_pos, (10, 35))
    status = font.render(f"Goals Remaining: {len(warehouse.goals)} | Score: {robot.score}", True, BLACK)
    screen.blit(status, (10, 60))
    
    # Show completion message
    if len(warehouse.goals) == 0:
        victory_text = font.render("MISSION COMPLETE! All packages delivered!", True, GREEN)
        screen.blit(victory_text, (300, 300))
    
    # Update display
    pygame.display.flip()

# Cleanup
pygame.quit()
sys.exit()
