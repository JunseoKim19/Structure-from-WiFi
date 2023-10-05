import cv2
import numpy as np
import pygame

map_image = cv2.imread('floorplans/testroom.png', cv2.IMREAD_GRAYSCALE)

width = 800
height = 800

map_image = cv2.resize(map_image, (width, height))

map_image = 255 - map_image

occupancy_grid = -1 * np.ones_like(map_image)

robot_position = np.array([700, 700])

grid_size = 10

movements = {
    
    pygame.K_w: np.array([0, -grid_size]),
    pygame.K_a: np.array([-grid_size, 0]),
    pygame.K_s: np.array([0, 0]),
    pygame.K_d: np.array([grid_size, 0]),
    pygame.K_x: np.array([0, grid_size])
}

rotation_matrices = []
for angle in np.linspace(0, 360, 45):
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    rotation_matrices.append(rotation_matrix)

def move_robot(current_position, key, map_image):
    if key in movements:
        move = movements[key]
        new_position = current_position + move
        new_position_int = new_position.astype(int)
        if map_image[new_position_int[0], new_position_int[1]] != 255:
            return new_position
    return current_position


log_odds_free = np.log(0.2 / 0.8)
log_odds_occupied = np.log(0.8 / 0.2)

def update_grid(grid, position, map_image):
    updated_cells = set()  # Keep track of which cells have been updated
    for rotation_matrix in rotation_matrices:
        hit_wall = False
        for i in range(1, 900):
            sensor_direction = np.dot(rotation_matrix, np.array([1, 0]))
            sensor_position = position + i * sensor_direction
            sensor_position_cell = (sensor_position // grid_size).astype(int)
            sensor_position_int = sensor_position.astype(int)

            if 0 <= sensor_position_cell[0] < grid.shape[0] and 0 <= sensor_position_cell[1] < grid.shape[1]:
                cell_key = tuple(sensor_position_cell)  # Convert to tuple so it can be used as a set key
                if cell_key not in updated_cells:  # Only update the cell if it hasn't been updated yet
                    updated_cells.add(cell_key)
                    if map_image[sensor_position_int[0], sensor_position_int[1]] == 255:
                        if grid[sensor_position_cell[0], sensor_position_cell[1]] > 0:
                            grid[sensor_position_cell[0], sensor_position_cell[1]] = max(0, grid[sensor_position_cell[0], sensor_position_cell[1]] - 15)
                        hit_wall = True
                    else:
                        if not hit_wall:
                            grid[sensor_position_cell[0], sensor_position_cell[1]] = min(255, grid[sensor_position_cell[0], sensor_position_cell[1]] + 15)
                        if hit_wall:
                            break
    return grid


occupancy_grid = 127 * np.ones((height // grid_size, width // grid_size), dtype=np.float32)

def grid_to_rgb(grid):
    grid_rgb = np.zeros_like(grid, dtype=np.uint8)
    grid_rgb[grid == 127] = 127 # Unknown
    grid_rgb[grid > 127] = grid[grid > 127] # Known area
    grid_rgb[grid < 127] = grid[grid < 127] # Wall

    grid_rgb = cv2.resize(grid_rgb, (width, height), interpolation=cv2.INTER_NEAREST)

    return cv2.cvtColor(grid_rgb, cv2.COLOR_GRAY2RGB)


pygame.init()

rgb_grid = grid_to_rgb(occupancy_grid)

grid_surface = pygame.Surface((rgb_grid.shape[1], rgb_grid.shape[0]))

window = pygame.display.set_mode((width, height))

robot_surface = pygame.Surface((10, 10))
robot_surface.fill((0, 0, 255))

try:
    while True:
        # Event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            elif event.type == pygame.KEYDOWN:
                if event.key in movements:
                    new_position = move_robot(robot_position, event.key, map_image)  # Pass map_image to move_robot
                    direction = new_position - robot_position
                    robot_position = new_position
                    occupancy_grid = update_grid(occupancy_grid, robot_position, map_image)
                    rgb_grid = grid_to_rgb(occupancy_grid)
                    pygame.surfarray.blit_array(grid_surface, rgb_grid)
        window.blit(grid_surface, (0, 0))
        window.blit(robot_surface, robot_position)
        grid_size = 10
        
        for x in range(0, window.get_width(), grid_size):
            pygame.draw.line(window, (255, 255, 255), (x, 0), (x, window.get_height()))
        
        for y in range(0, window.get_height(), grid_size):
            pygame.draw.line(window, (255, 255, 255), (0, y), (window.get_width(), y))

        pygame.display.flip()
except KeyboardInterrupt:
    pass