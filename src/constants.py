"""
Constants for the warehouse robot simulation.
"""

# Screen dimensions
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60
GRID_SIZE = 40

# Warehouse grid dimensions
WAREHOUSE_WIDTH = SCREEN_WIDTH // GRID_SIZE
WAREHOUSE_HEIGHT = SCREEN_HEIGHT // GRID_SIZE

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
RED_BOX_COLOR = (255, 0, 0)  # Bright red for high-priority cargo

# Red box settings
RED_BOX_EXPIRATION_TIME = 7000  # 7 seconds in milliseconds

# Debugging flag
DEBUG = True

def debug_log(message):
    """Print debug message if debugging is enabled."""
    if DEBUG:
        print(f"[MAIN DEBUG] {message}")

