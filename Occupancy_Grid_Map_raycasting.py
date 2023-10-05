import cv2
import numpy as np
import pygame

map_image = cv2.imread('floorplans/testroom4.png', cv2.IMREAD_GRAYSCALE)
print("Unique values in map_image:", np.unique(map_image))
width = 800
height = 800
map_image = cv2.resize(map_image, (width, height))

occupancy_grid = -1 * np.ones_like(map_image)
grid_size = 10

robot_position = np.array([750, 750])

transmitter_position = robot_position

movement_speed = 10
router_points = np.array([[600, 600]])
num_routers = len(router_points)
#[590, 300]
receiver_positions = router_points

trajectory = np.array([[750, 750], [750, 350], [590, 350], [590, 310], [750, 310], [750, 40], [40, 40], [40, 310], [590, 310], [590, 350], [420, 350], [420, 440], [380, 440], [380, 350], [40, 350], [40, 580], [200, 580], [200, 630], [40, 630], [40, 750], [380, 750], [380, 630], [200, 630], [200, 580], [380, 580], [380, 440], [420, 440], [420, 750], [750, 750]])


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

def count_intersections(map_image, start, end):
    points = list(supercover_line(start[0], start[1], end[0], end[1]))
    num_intersections = 0
    on_wall = False

    for p in points:
        if map_image[int(p[1]), int(p[0])] == 0:  # this is a wall
            if not on_wall:  # it's a new wall
                num_intersections += 1
                on_wall = True
        else:
            on_wall = False  # reset on_wall as we've hit a non-wall cell

    return num_intersections

def estimate_walls(map_image, start, end):
    num_walls = count_intersections(map_image, start, end)
    print("Estimated number of walls:", num_walls)
    return num_walls

n = 2
A = -30
d0 = 1
wall_loss = 30
L_w = 50

def calculate_rssi(transmitter_position, receiver_position, n, A, d0, num_walls):
    print("Transmitter position:", transmitter_position)
    print("Initial robot position:", robot_position)
    print("Receiver position:", receiver_position)
    distance = np.linalg.norm(transmitter_position - receiver_position)
    distance += 1e-10
    rssi = -10 * n * np.log10(distance / d0) + A - L_w * num_walls
    print("Calculated RSSI:", rssi)
    return rssi


def update_grid(grid, start, end, wall_loss, num_walls, robot_trajectory):
    points = list(supercover_line(start[0], start[1], end[0], end[1]))
    k_points_indices = []

    # Identify the intersection points
    for idx, p in enumerate(points):
        if map_image[int(p[1]), int(p[0])] == 0:
            k_points_indices.append(idx)

    # Handle the k = 0 case (no walls)
    if num_walls == 0:
        for p in points:
            sensor_position = np.array(p)
            sensor_position_cell = (sensor_position // grid_size).astype(int)
            grid[sensor_position_cell[0], sensor_position_cell[1]] = min(255, grid[sensor_position_cell[0], sensor_position_cell[1]] + 30)

    # Handle the k = 1 case (one wall)
    elif num_walls == 1:
        if len(k_points_indices) == 1:
            # Only one green point, indicating a wall
            for p in points:
                sensor_position = np.array(p)
                sensor_position_cell = (sensor_position // grid_size).astype(int)
                if grid[sensor_position_cell[0], sensor_position_cell[1]] <= 127:
                    grid[sensor_position_cell[0], sensor_position_cell[1]] = max(0, grid[sensor_position_cell[0], sensor_position_cell[1]] - 30)
        else:
            # Multiple green points, indicating no wall between them
            for i in range(1, len(k_points_indices)):
                start_idx = k_points_indices[i-1]
                end_idx = k_points_indices[i]
                for j in range(start_idx + 1, end_idx):
                    sensor_position = np.array(points[j])
                    sensor_position_cell = (sensor_position // grid_size).astype(int)
                    grid[sensor_position_cell[0], sensor_position_cell[1]] = min(255, grid[sensor_position_cell[0], sensor_position_cell[1]] + 30)

    # Handle the k = 2 case (two walls)
    elif num_walls == 2:
        # Similar logic to k=1, but for two walls
        if len(k_points_indices) == 2:
            for p in points:
                sensor_position = np.array(p)
                sensor_position_cell = (sensor_position // grid_size).astype(int)
                if grid[sensor_position_cell[0], sensor_position_cell[1]] <= 127:
                    grid[sensor_position_cell[0], sensor_position_cell[1]] = max(0, grid[sensor_position_cell[0], sensor_position_cell[1]] - 30)
        else:
            for i in range(1, len(k_points_indices), 2):
                start_idx = k_points_indices[i-1]
                end_idx = k_points_indices[i]
                for j in range(start_idx + 1, end_idx):
                    sensor_position = np.array(points[j])
                    sensor_position_cell = (sensor_position // grid_size).astype(int)
                    grid[sensor_position_cell[0], sensor_position_cell[1]] = min(255, grid[sensor_position_cell[0], sensor_position_cell[1]] + 30)
    
    for point in robot_trajectory:
        traj_cell = (np.array(point) // grid_size).astype(int)
        if 0 <= traj_cell[0] < grid.shape[0] and 0 <= traj_cell[1] < grid.shape[1]:
            grid[traj_cell[0], traj_cell[1]] = 255

    return grid

# Now, you can execute the main loop as you had it before to see the changes.



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
window_width = 2 * width
window = pygame.display.set_mode((window_width, height))

robot_surface = pygame.Surface((10, 10))
robot_surface.fill((0, 0, 255))

map_surface = pygame.surfarray.make_surface(map_image)
map_surface = pygame.transform.scale(map_surface, (width, height)) 
map_surface = pygame.transform.rotate(map_surface, -90)
map_surface = pygame.transform.flip(map_surface, True, False)

robot_trajectory = [robot_position.tolist()]
walls_per_segment = []  # To store number of walls for each segment

running = True

try:
    for target_position in trajectory:
        direction = target_position - robot_position
        direction_norm = np.linalg.norm(direction)
        if direction_norm != 0:
            step = direction / direction_norm * movement_speed
        else:
            step = direction

        while np.linalg.norm(robot_position - target_position) > movement_speed:
            robot_position = robot_position.astype(float) + step
            robot_trajectory.append(robot_position.tolist())
            
            for router in router_points:
                num_walls = estimate_walls(map_image, robot_position, router)
                walls_per_segment.append(num_walls)  # Add to the list

                rssi = calculate_rssi(robot_position, router, n, A, d0, num_walls)
                occupancy_grid = update_grid(occupancy_grid, robot_position, router, wall_loss, num_walls, robot_trajectory)

            robot_position_int = robot_position.astype(int)  # Moved this line outside the inner while loop

            rgb_grid = grid_to_rgb(occupancy_grid)
            pygame.surfarray.blit_array(grid_surface, rgb_grid)

            window.blit(grid_surface, (0, 0))
            window.blit(robot_surface, tuple(robot_position_int))
            for point in router_points:
                pygame.draw.circle(window, (255, 0, 0), tuple(point), 10)

            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
            for i in range(1, len(robot_trajectory)):
                color = colors[walls_per_segment[i-1] if i-1 < len(walls_per_segment) else 0]
                pygame.draw.line(window, color, robot_trajectory[i-1], robot_trajectory[i], 2)

            for i in range(0, width, grid_size):
                pygame.draw.line(window, (200, 200, 200), (i, 0), (i, height))
            for i in range(0, height, grid_size):
                pygame.draw.line(window, (200, 200, 200), (0, i), (width, i))

            window.blit(map_surface, (width, 0))
            pygame.display.update()
            pygame.time.delay(100)

except StopIteration:
    running = False

if running:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False