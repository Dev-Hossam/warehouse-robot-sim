# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Quick Start Commands

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Simulation
```bash
python src/main.py
```

## Project Overview

**Warehouse Robot Simulation** is a pygame-based simulation framework for modeling robot behavior in a warehouse environment. The codebase is currently in early stages with a minimal pygame window setup.

### Architecture

The project uses a simple monolithic structure:
- **`src/main.py`**: Entry point that initializes pygame, sets up the display (800x600), manages the game loop at 60 FPS, and handles basic event processing (quit on ESC or window close)

### Key Components

- **pygame**: Core simulation engine for graphics and event handling
- **Main Loop**: Handles FPS regulation, event processing, and screen rendering
- **Display**: 800x600 window with white background

## Development Notes

- The simulation currently runs a basic game loop without any robot logic or warehouse environment simulation yet
- The codebase is minimal and ready for expansion with robot classes, warehouse grid system, pathfinding logic, and task management
- All code is contained in a single file; consider modularizing into separate components as complexity grows (e.g., `robots.py`, `warehouse.py`, `physics.py`)
