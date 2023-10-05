import cv2
import numpy as np
import pygame

map_image = cv2.imread('floorplans/testroom.png', cv2.IMREAD_GRAYSCALE)

width = 800
height = 800

map_image = cv2.resize(map_image, (width, height))

map_image = 255 - map_image

occupancy_grid = -1 * np.ones_like(map_image)

robot_position = np.array([100, 650])
#robot_position = np.array([map_image.shape[0] // 2, map_image.shape[1] // 2]) #Middle Point

movements = {
    pygame.K_w: np.array([0, -1]),  # Up
    pygame.K_a: np.array([-1, 0]),  # Left
    pygame.K_s: np.array([0, 0]),   # Stop
    pygame.K_d: np.array([1, 0]),   # Right
    pygame.K_x: np.array([0, 1])    # Down
}

def move_robot(current_position, key):
    if key in movements:
        move = movements[key]
        new_position = current_position + move
        return new_position
    else:
        return current_position

rotation_matrices = []
for angle in np.linspace(0, 1500, 1500):
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
    rotation_matrices.append(rotation_matrix)

occupancy_grid = np.zeros_like(map_image, dtype=np.float32)

log_odds_free = np.log(0.3 / 0.7)
log_odds_occupied = np.log(0.7 / 0.3)

def update_grid(grid, position, map_image, k_visibility_grid):
    for rotation_matrix in rotation_matrices:
        wall_counter = 0
        inside_wall = False
        for i in range(1, 950):
            sensor_direction = np.dot(rotation_matrix, np.array([1, 0]))
            sensor_position = position + i * sensor_direction
            sensor_position = sensor_position.astype(int)
            if 0 <= sensor_position[0] < grid.shape[0] and 0 <= sensor_position[1] < grid.shape[1]:
                if map_image[sensor_position[0], sensor_position[1]] == 255:
                    if not inside_wall:
                        inside_wall = True
                else:
                    if inside_wall:
                        wall_counter += 1
                        inside_wall = False
                k_visibility_grid[sensor_position[0], sensor_position[1]] = min(wall_counter, 3)
                grid[sensor_position[0], sensor_position[1]] += log_odds_occupied if wall_counter == 0 else log_odds_free
    return grid, k_visibility_grid



def grid_to_rgb(grid, k_visibility_grid):
    rgb_grid = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    rgb_grid[k_visibility_grid == 0] = [255, 0, 0]  # Red for k=0
    rgb_grid[k_visibility_grid == 1] = [0, 255, 0]  # Green for k=1
    rgb_grid[k_visibility_grid == 2] = [0, 0, 255]  # Blue for k=2
    rgb_grid[k_visibility_grid == 3] = [255, 255, 0]  # Yellow for k=3
    rgb_grid[k_visibility_grid == -1] = [200, 200, 200]  # White for free space
    rgb_grid[map_image == 255] = [0, 0, 0]  # Black for walls
    return rgb_grid



pygame.init()

k_visibility_grid = np.full_like(occupancy_grid, fill_value=-1, dtype=np.int32)
rgb_grid = grid_to_rgb(occupancy_grid, k_visibility_grid)

grid_surface = pygame.Surface((rgb_grid.shape[0], rgb_grid.shape[1]))

window = pygame.display.set_mode((width, height))

robot_surface = pygame.Surface((10, 10))
robot_surface.fill((0, 0, 255))

try:
    while True:
        # Event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        for key in movements:
            if keys[key]:
                new_position = move_robot(robot_position, key)
                direction = new_position - robot_position
                robot_position = new_position
                k_visibility_grid = np.full_like(occupancy_grid, fill_value=-1, dtype=np.int32)
                occupancy_grid, k_visibility_grid = update_grid(occupancy_grid, robot_position, map_image, k_visibility_grid)
                rgb_grid = grid_to_rgb(occupancy_grid, k_visibility_grid)
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
