import cv2
import numpy as np
import pygame
from sklearn.svm import SVR
import heapq

# Load the map image
map_image = cv2.imread('floorplans/testroom.png', cv2.IMREAD_GRAYSCALE)
print("Unique values in map_image:", np.unique(map_image))
# Resize the map image
width = 800
height = 800
map_image = cv2.resize(map_image, (width, height))

# Invert the map image
map_image = 255 - map_image

# Initialize the occupancy grid
occupancy_grid = -1 * np.ones_like(map_image)
grid_size = 10

# Set the initial position of the robot
robot_position = np.array([700, 700])
print("Initial robot position:", robot_position)
transmitter_position = robot_position

# Define the robot's movement speed
movement_speed = 5

# Define the number of routers
num_routers = 5

# Generate random x and y coordinates for the routers
router_x_coords = np.random.randint(0, width, size=num_routers)
router_y_coords = np.random.randint(0, height, size=num_routers)

# Combine the x and y coordinates to create the router positions
router_points = np.column_stack((router_x_coords, router_y_coords))

receiver_positions = router_points

# Define the path loss model parameters
n = 2
A = -30
d0 = 1


# Define the wall prediction model
wall_prediction_model = SVR()

def calculate_rssi(transmitter_position, receiver_position, n, A, d0):
    print("Transmitter position:", transmitter_position)
    print("Receiver position:", receiver_position)
    print("n, A, d0:", n, A, d0)
    distance = np.linalg.norm(transmitter_position - receiver_position)
    distance += 1e-10
    rssi = -10 * n * np.log10(distance / d0) + A
    print("Calculated RSSI:", rssi)
    return rssi

def a_star(map_image, start, end):
    # Convert start and end points to tuples
    start = tuple(start)
    end = tuple(end)
    movements = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    closed_nodes = set()
    node_costs = {start: 0}
    paths = {start: []}
    open_nodes = [(0, start)]
    while open_nodes:
        current_cost, current_node = heapq.heappop(open_nodes)
        if current_node in closed_nodes:
            continue
        if current_node == end:
            return paths[current_node]
        closed_nodes.add(current_node)
        for dx, dy in movements:
            next_node = (current_node[0] + dx, current_node[1] + dy)
            if not (0 <= next_node[0] < map_image.shape[0] and 0 <= next_node[1] < map_image.shape[1]):
                continue
            if map_image[next_node] == 255:
                continue
            next_cost = current_cost + 1
            if next_node not in node_costs or next_cost < node_costs[next_node]:
                node_costs[next_node] = next_cost
                paths[next_node] = paths[current_node] + [current_node]
                heapq.heappush(open_nodes, (next_cost + np.linalg.norm(np.array(end) - np.array(next_node)), next_node))
    return None

X_train = []
y_train = []
print("X_train:", X_train)
print("y_train:", y_train)

for x in range(0, width, grid_size):
    for y in range(0, height, grid_size):
        position = np.array([x, y])

        rssi = calculate_rssi(transmitter_position, position, n, A, d0)

        path = a_star(map_image, position, position)
        num_walls = sum(map_image[node] == 255 for node in path)

        X_train.append(rssi)
        y_train.append(num_walls)

X_train = np.array(X_train).reshape(-1, 1)
y_train = np.array(y_train)

print("X_train NaN values:", np.isnan(X_train).any())
print("y_train NaN values:", np.isnan(y_train).any())
wall_prediction_model.fit(X_train, y_train)

def update_grid(grid, path, grid_size):
    updated_cells = set() 

    for p in path:
        sensor_position = np.array(p)
        sensor_position_cell = (sensor_position // grid_size).astype(int)

        if 0 <= sensor_position_cell[0] < grid.shape[0] and 0 <= sensor_position_cell[1] < grid.shape[1]:
            cell_key = tuple(sensor_position_cell)  # Convert to tuple so it can be used as a set key
            if cell_key not in updated_cells:  # Only update the cell if it hasn't been updated yet
                updated_cells.add(cell_key)

                if map_image[sensor_position_cell[0], sensor_position_cell[1]] == 255:
                    # If the cell is a wall, make the cell darker
                    grid[sensor_position_cell[0], sensor_position_cell[1]] = max(0, grid[sensor_position_cell[0], sensor_position_cell[1]] - 5)
                else:
                    # If the cell is not a wall, make the cell whiter
                    grid[sensor_position_cell[0], sensor_position_cell[1]] = min(255, grid[sensor_position_cell[0], sensor_position_cell[1]] + 5)

    return grid

occupancy_grid = 127 * np.ones((height // grid_size, width // grid_size), dtype=np.float32)

def grid_to_rgb(grid):
    grid_rgb = np.zeros_like(grid, dtype=np.uint8)
    grid_rgb[grid == 127] = 127 # Unknown
    grid_rgb[grid > 127] = grid[grid > 127] # Known area
    grid_rgb[grid < 127] = grid[grid < 127] # Wall

    grid_rgb = cv2.resize(grid_rgb, (width, height), interpolation=cv2.INTER_NEAREST)

    return cv2.cvtColor(grid_rgb, cv2.COLOR_GRAY2RGB)

# Initialize Pygame
pygame.init()
# Create a surface for the grid
rgb_grid = grid_to_rgb(occupancy_grid)
grid_surface = pygame.Surface((rgb_grid.shape[1], rgb_grid.shape[0]))

# Create a window
window = pygame.display.set_mode((2*width, height))

# Create a surface for the robot
robot_surface = pygame.Surface((10, 10))
robot_surface.fill((0, 0, 255))

# Set the current waypoint index
current_waypoint_index = 0
print("Wall prediction model output for sample data:", wall_prediction_model.predict([[0], [1], [2]]))

# Create a second window for the ground truth
#truth_window = pygame.display.set_mode((width, height), pygame.RESIZABLE, 1)
truth_window_surface = pygame.Surface((width, height))
# Convert the map image to a Pygame surface
map_surface = pygame.surfarray.make_surface(map_image)
# Store the last position where A* was run
last_astar_position = None
# Define the minimum distance before running A* again
min_astar_distance = 10  # Adjust this value as needed

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    robot_position[1] -= movement_speed  # Move up
                elif event.key == pygame.K_DOWN:
                    robot_position[1] += movement_speed  # Move down
                elif event.key == pygame.K_LEFT:
                    robot_position[0] -= movement_speed  # Move left
                elif event.key == pygame.K_RIGHT:
                    robot_position[0] += movement_speed  # Move right

        new_position = robot_position.astype(int)
        path_is_clear = True

        # Only run A* if the robot has moved more than the minimum distance or if it's the first run
        if last_astar_position is None or np.linalg.norm(robot_position - last_astar_position) >= min_astar_distance:
            for router in router_points:
                # Calculate RSSI for each router
                rssi = calculate_rssi(robot_position, router, n, A, d0)

                # Estimate the number of walls for each router
                path = a_star(map_image, robot_position, router)
                if path is not None:
                    num_walls = sum(map_image[node] == 255 for node in path)
                    occupancy_grid = update_grid(occupancy_grid, path, grid_size)
                else:
                    print(f"No path found from {robot_position} to {router}")

            # Store the current position as the last A* position
            last_astar_position = robot_position.copy()

        rgb_grid = grid_to_rgb(occupancy_grid)
        pygame.surfarray.blit_array(grid_surface, rgb_grid)

        window.blit(grid_surface, (0, 0))
        window.blit(robot_surface, robot_position)
        for point in router_points:
            pygame.draw.circle(window, (255, 0, 0), point, 10)  # Draw a red circle at each router point

        # Update the ground truth surface
        truth_window_surface.blit(map_surface, (0, 0))  # Draw the map on the truth window surface
        truth_window_surface.blit(robot_surface, (new_position[0]-5, new_position[1]-5))  # Draw the robot on the truth window surface
        for point in router_points:
            pygame.draw.circle(truth_window_surface, (255, 0, 0), point, 10)  # Draw the routers on the truth window surface

        window.blit(truth_window_surface, (width, 0))  # Blit the truth window surface on the right half of the main window

        pygame.display.update() # Update the window

        pygame.time.delay(1)
except KeyboardInterrupt:
    pass

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()



