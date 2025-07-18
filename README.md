# Trajectory Planning Package

A comprehensive ROS2 package for autonomous vehicle trajectory planning and path following, specifically designed for F1Tenth racing cars.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Nodes](#nodes)
- [Launch Files](#launch-files)
- [Parameters](#parameters)
- [Contributing](#contributing)
- [License](#license)
- [Maintainers](#maintainers)

## Overview

This package provides a complete trajectory planning and path following system for autonomous vehicles. It includes multiple path planning algorithms (A*, Dijkstra), lookahead-based controllers, and pure pursuit path following capabilities.

## Features

- **Multiple Path Planning Algorithms**:
  - A* pathfinding with configurable heuristics
  - Dijkstra's algorithm for guaranteed optimal paths
  
- **Flexible Path Sources**:
  - CSV file-based racing line loading
  - Dynamic lookahead point generation
  - Real-time path planning integration

- **Advanced Control Systems**:
  - Pure pursuit controller with obstacle avoidance
  - Configurable lookahead distances
  - Speed-dependent control parameters

- **ROS2 Integration**:
  - Native ROS2 nodes with proper parameter handling
  - TF2 coordinate frame transformations
  - Action server integration with Nav2

- **Visualization Tools**:
  - RViz markers for path visualization
  - Lookahead circle and point display
  - Path trajectory rendering

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Path      │    │   A* Lookahead   │    │   A* Path       │
│   Publisher     │────│   Publisher      │────│   Planner       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Pure Pursuit  │◄───│   Lookahead to   │◄───│   Path          │
│   Controller    │    │   Planner        │    │   Planner       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

1. **Clone the repository**:
   ```bash
   cd ~/ros2_ws/src
   git clone <repository-url>
   ```

2. **Install dependencies**:
   ```bash
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. **Build the package**:
   ```bash
   colcon build --packages-select trajectory_planning
   ```

4. **Source the workspace**:
   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

## Configuration

All configuration parameters are stored in YAML files in the `config/` directory:

- `a_star_planner_config.yaml` - A* algorithm parameters
- `astar_lookahead_pub_config.yaml` - Lookahead publisher settings
- `csv_pub_config.yaml` - CSV path publisher configuration
- `lookahead_to_planner_config.yaml` - Planner bridge settings
- `pure_pursuit_v2_params.yaml` - Pure pursuit controller parameters

### Key Configuration Parameters

#### A* Planner
```yaml
a_star_node:
  ros__parameters:
    is_antiClockwise: false
    costmap_topic: "/costmap/costmap"
    path_topic: "/pp_path"
```

#### Pure Pursuit Controller
```yaml
pure_pursuit_v2:
  ros__parameters:
    pure_pursuit_lookahead_distance: 0.18  # meters
    pure_pursuit_max_drive_speed: 0.1      # m/s
    pure_pursuit_wheel_base: 0.16          # meters
```

## Usage

### Quick Start

Launch the complete trajectory planning system:
```bash
ros2 launch trajectory_planning complete_trajectory_planning.launch.py
```

### Individual Components

Launch specific components as needed:

```bash
# A* Path Planner
ros2 launch trajectory_planning a_star_planner.launch.py

# CSV Racing Line Publisher
ros2 launch trajectory_planning csv_racingline_publisher.launch.py

# Pure Pursuit Controller
ros2 launch trajectory_planning pure_pursuit_v2_launch.py
```

### Manual Node Execution

Run individual nodes with custom parameters:

```bash
# A* Lookahead Publisher
ros2 run trajectory_planning astar_lookahead_pub_exe

# CSV Path Publisher
ros2 run trajectory_planning csv_pub_exe

# Pure Pursuit Controller
ros2 run trajectory_planning pure_pursuit_node_v2
```

## Nodes

### astar_lookahead_pub_node
Publishes lookahead points based on robot position and CSV waypoints.

**Subscribed Topics:**
- `/tf` (TF transformations)

**Published Topics:**
- `/astar_lookahead_marker` (Visualization markers)
- `/astar_lookahead_circle` (Lookahead circle visualization)

### csv_path_pub
Loads and publishes racing lines from CSV files.

**Published Topics:**
- `/csv_pp_path` (Path messages)

### a_star_node
Implements A* path planning algorithm.

**Subscribed Topics:**
- `/costmap/costmap` (Occupancy grid)
- `/astar_lookahead_marker` (Goal markers)

**Published Topics:**
- `/pp_path` (Planned paths)

### lookahead_to_path_planner
Bridges lookahead markers with Nav2 path planning.

**Subscribed Topics:**
- `/astar_lookahead_marker` (Goal markers)

**Published Topics:**
- `/astar_pp_path` (Computed paths)

**Action Clients:**
- `/compute_path_to_pose` (Nav2 path planning)

### pure_pursuit_v2
Advanced pure pursuit controller with obstacle avoidance.

**Subscribed Topics:**
- `/ego_racecar/odom` (Robot odometry)
- `/map` (Occupancy grid)
- `/pure_pursuit/path` (Target path)
- `/pure_pursuit/enabled` (Enable/disable control)

**Published Topics:**
- `/cmd_vel` (Velocity commands)
- `/pure_pursuit/lookahead` (Lookahead point)

## Launch Files

| Launch File | Description |
|-------------|-------------|
| `complete_trajectory_planning.launch.py` | Launches entire trajectory planning system |
| `a_star_planner.launch.py` | A* path planner only |
| `astar_lookahead_pub.launch.py` | Lookahead publisher only |
| `csv_racingline_publisher.launch.py` | CSV path publisher only |
| `dijkstra_planner.launch.py` | Dijkstra path planner only |
| `lookahead_to_planner.launch.py` | Planner bridge only |
| `pure_pursuit_v2_launch.py` | Pure pursuit controller only |

## Parameters

### Global Parameters

- **Frame IDs**: Configure coordinate frames for transformations
- **Topic Names**: Customize ROS topic names for integration
- **Algorithm Settings**: Tune path planning and control algorithms

### Path Planning Parameters

- **Lookahead Distance**: Distance for lookahead point calculation
- **Search Directions**: 4-connected vs 8-connected grid search
- **Cost Weights**: Penalty weights for different terrain types

### Control Parameters

- **Wheel Base**: Vehicle wheelbase for kinematic calculations
- **Speed Limits**: Maximum linear and angular velocities
- **Obstacle Avoidance**: Field-of-view and safety margins

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Add comprehensive docstrings to all functions and classes
- Use type hints where appropriate
- Include unit tests for new functionality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Maintainers

- **Fam Shihata** - [fam@awadlouis.com](mailto:fam@awadlouis.com)
- **George Halim**

## Acknowledgments

- F1Tenth Community for the autonomous racing platform
- ROS2 Development Team for the robotics framework
- Contributors to the Nav2 navigation stack

## Version History

- **v0.2.3** - Professional refactoring with comprehensive documentation
- **v0.2.0** - Added pure pursuit controller and multiple path planners
- **v0.1.0** - Initial release with basic A* planning

---

For more information, please visit our [documentation](docs/) or contact the maintainers.