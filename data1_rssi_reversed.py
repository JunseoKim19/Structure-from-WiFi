import numpy as np
import pygame
import pandas as pd
import cv2

data_path = "rssi_data4.csv"
data = pd.read_csv(data_path)
x_pixel = data['x_pixel'].values
y_pixel = data['y_pixel'].values
rssi_values = data['rssi'].values
trajectory_from_csv = list(zip(x_pixel, y_pixel))

width = 700
height = 700

grid_size = 5

robot_position = np.array([trajectory_from_csv[0][0], trajectory_from_csv[0][1]])

transmitter_position = robot_position

movement_speed = 2
router_points = [np.array([379 , 236])]

receiver_positions = router_points

trajectory = list(zip(x_pixel, y_pixel))
trajectory_grid = np.zeros((height // grid_size, width // grid_size, 3), dtype=np.uint8)

def supercover_line(x0, y0, x1, y1):
    points = []
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(int(x1 - x0))
    dy = abs(int(y1 - y0))

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        points.append((x0 + x*xx + y*yx, y0 + x*xy + y*yy))
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

    return points

def update_grid(grid, start, end, rssi):
    updated_cells = set()
    points = list(supercover_line(start[0], start[1], end[0], end[1]))
    robot_position_cell = (np.array(start) // grid_size).astype(int)

    if rssi >= -37:
        k_visibility = 0
    elif -37> rssi >= -45.001:
        k_visibility = 1
    else:
        k_visibility = 2

    robot_color = trajectory_color_map[k_visibility]
    trajectory_grid[robot_position_cell[0], robot_position_cell[1]] = robot_color

    k_points_indices = []

    for idx, p in enumerate(points):
        sensor_position = np.array(p)
        sensor_position_cell = (sensor_position // grid_size).astype(int)
        if 0 <= sensor_position_cell[0] < grid.shape[0] and 0 <= sensor_position_cell[1] < grid.shape[1]:
            cell_key = tuple(sensor_position_cell)
            if cell_key not in updated_cells:
                updated_cells.add(cell_key)
                if np.array_equal(trajectory_grid[sensor_position_cell[0], sensor_position_cell[1]], robot_color):
                    k_points_indices.append(idx)
                    continue

    if k_visibility == 0:
        for p in points:
            sensor_position = np.array(p)
            sensor_position_cell = (sensor_position // grid_size).astype(int)
            grid[sensor_position_cell[0], sensor_position_cell[1]] = min(255, grid[sensor_position_cell[0], sensor_position_cell[1]] + 30)
    else:
        if len(k_points_indices) <= 1:
            for p in points:
                sensor_position = np.array(p)
                sensor_position_cell = (sensor_position // grid_size).astype(int)
                if grid[sensor_position_cell[0], sensor_position_cell[1]] <= 127:
                    grid[sensor_position_cell[0], sensor_position_cell[1]] = max(0, grid[sensor_position_cell[0], sensor_position_cell[1]] - 30)
        else:
            for i in range(1, len(k_points_indices)):
                start_idx = k_points_indices[i-1]
                end_idx = k_points_indices[i]
                for j in range(start_idx + 1, end_idx):
                    sensor_position = np.array(points[j])
                    sensor_position_cell = (sensor_position // grid_size).astype(int)
                    grid[sensor_position_cell[0], sensor_position_cell[1]] = min(255, grid[sensor_position_cell[0], sensor_position_cell[1]] + 30)
    return grid


occupancy_grid = 127 * np.ones((height // grid_size, width // grid_size), dtype=np.float32)

def grid_to_rgb(grid):
    grid_rgb_pre = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
    grid_rgb = cv2.resize(grid_rgb_pre, (width, height), interpolation=cv2.INTER_NEAREST)
    
    trajectory_resized = cv2.resize(trajectory_grid, (width, height), interpolation=cv2.INTER_NEAREST)
    
    mask = (trajectory_resized != [0, 0, 0]).all(axis=2)
    grid_rgb[mask] = trajectory_resized[mask]
    
    return grid_rgb

def execute_trajectory(trajectory, rssi_values, robot_position):
    global occupancy_grid
    for target_position, rssi in zip(trajectory, rssi_values):
        if rssi >= -37:
            k_visibility = 0
        elif -37 > rssi >= -45.001:
            k_visibility = 1
        else:
            k_visibility = 2
        direction = np.array(target_position) - robot_position
        direction_norm = np.linalg.norm(direction)
        if direction_norm != 0:
            step = direction / direction_norm * movement_speed
        else:
            step = direction

        while np.linalg.norm(robot_position - np.array(target_position)) > movement_speed:
            robot_position = robot_position.astype(float) + step
            robot_trajectory.append((robot_position.tolist(), k_visibility))
            robot_position_int = robot_position.astype(int)
            
            for router in router_points:
                occupancy_grid = update_grid(occupancy_grid, robot_position, router, rssi)
                
            rgb_grid = grid_to_rgb(occupancy_grid)
            pygame.surfarray.blit_array(window, rgb_grid)

            for i in range(0, width, grid_size):
                pygame.draw.line(window, (200, 200, 200), (i, 0), (i, height))
            for i in range(0, height, grid_size):
                pygame.draw.line(window, (200, 200, 200), (0, i), (width, i))
            for i in range(1, len(robot_trajectory)):
                color = trajectory_color_map[robot_trajectory[i][1]]
                pygame.draw.line(window, color, robot_trajectory[i-1][0], robot_trajectory[i][0], 2)

            window.blit(robot_surface, tuple(robot_position_int))
            for point in router_points:
                window.blit(router_surface, tuple(point))
            
            pygame.display.update()
            pygame.time.delay(100)
    return robot_position

pygame.init()

window = pygame.display.set_mode((width, height))
robot_surface = pygame.Surface((10, 10))
robot_surface.fill((0, 0, 255))
router_surface = pygame.Surface((10, 10))
router_surface.fill((255, 0, 0))

trajectory_color_map = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)  
}

robot_trajectory = [(robot_position.tolist(), 0)]

previous_rssi = rssi_values[0]

robot_position = execute_trajectory(trajectory, rssi_values, robot_position)

trajectory = list(reversed(trajectory))
rssi_values = list(reversed(rssi_values))

robot_position = execute_trajectory(trajectory, rssi_values, robot_position)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
