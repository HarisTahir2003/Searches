# %% [markdown]
# # <center> <strong> Introduction to Artificial Intelligence </strong> </center>
# # <center> <strong> Haris Tahir Rana </strong> </center>

# %% [markdown]
# ### <b> Problem Description </b>
# 
# In this assignment, we are tasked with navigating through the prestigious LUMS University using three search algorithms: 
# <ol>
# <li> A* </li>
# <li> Best-first search </li>
# <li> Breadth-first search </li>
# </ol> <br>
# For the purpose of this assignment, we will be walking through "legal" paths, i.e. paths that do not involve any shortcuts or walking through grounds etc. <br/>
# 
# Please read through the below **starter code** (please read comments and use of the functions as these will be used later in your main function, or when you wish to test your implementations). <br/>
# 
#  We have also provided formulas for the different heuristic functions that need to be implemented and used.
# 
# 
# 

# %% [markdown]
# ### Import Libraries

# %%
import osmnx as ox
from IPython.display import IFrame
import networkx as nx
import folium
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import queue

# %% [markdown]
# ## Folium and osmnx: Interactive Geospatial Visualization and Analysis
# 
# Let us first learn about the functionality of some libraries used in the starter code.
# ## osmnx
# **osmnx** is a Python package that facilitates retrieving and visualizing data from OpenStreetMap.
# 
# In this assignment, we are using osmnx to:
# 
# - **Get Road Network Data**: Fetch road network data for LUMS from OpenStreetMap (OSM). This data includes information about roads, intersections, and other elements of the network.  <br/>
# 
# - **Plot Road Networks**: Plot road networks directly onto Folium maps.
# 
# ## Folium
# **Folium** is a Python library that facilitates the creation of interactive maps directly from Python code. It's particularly useful for visualizing geographical data.
# 
# In this assignment, we are using Folium to:
# 
# - **Visualize Road Networks**: Plot road networks obtained from OpenStreetMap (OSM) onto an interactive map. You can see the layout of roads, intersections, and other features. <br/>
# 
# - **Customize Markers**: Add markers to the map to highlight specific locations, such as nodes in the road network. These markers can display additional information when clicked, like node IDs or names. <br/>
# 
# - **Display Shortest Paths**: Visualize shortest paths on the map. You can see the route between two points highlighted on the map, along with the nodes and edges that form the path. <br/>
# 
# 

# %% [markdown]
# ### Road Network Data Retrieval
# 
# Note that we have set network_type to "walk" so all paths that can be walked according the map's definitions will be used. You can try other settings like "drive" and see how the path you get in the visualizations changes to just include all the roads in LUMS.

# %%
# Get the road network data from OpenStreetMap
def get_road_network(location, distance):
    G = ox.graph_from_address(location, network_type='walk', dist=distance)
    return G

# %% [markdown]
# ### Road Network Visualization Function

# %%
# Function to visualize the road network using Folium
def visualize_road_network(G, location, distance):
    """
        G: Graph object representing the road network
        location: location for which the road network is to be visualized
        distance: distance (in meters) around the specified location to be visualized
    """
    # Get the center (latitude and longitude) coordinates of the location
    lat, lng = ox.geocode(location)

    # Create a folium map centered at the location with an initial zoom level of 12
    map_center = [lat, lng]
    map_osm = folium.Map(location=map_center, zoom_start=12)

    # Add the road network graph (represented by "G") to the folium map
    # This overlays the road network onto the map, showing roads, intersections, and other features
    ox.plot_graph_folium(G, graph_map=map_osm, popup_attribute='name', node_labels=True, edge_width=20)

    # Add customized markers for nodes to view node IDs etc upon click
    for node, data in G.nodes(data=True):
        folium.Marker(location=[data['y'], data['x']], popup=f"Node: {node}").add_to(map_osm)

    # Display the folium map inline
    display(map_osm)

# %% [markdown]
# ### Shortest Path Visualization Function

# %%
# Function to visualize the shortest path on the map
def visualize_path_folium(G, shortest_path, location, source_node, target_nodes, distance):
    """
        G: Graph object representing the road network
        shortest_path:  A list of node IDs representing the shortest path between the source and target nodes
        location: location around which the map is centered
        source_node:  ID of the source node
        target_nodes: list of ID(s) of target nodes
        distance: distance (in meters) between the source and target nodes
    """

    # Get the center (latitude and longitude) coordinates of the location
    lat, lng = ox.geocode(location)

    # Create a folium map centered at the location
    map_center = [lat, lng]
    map_osm = folium.Map(location=map_center, zoom_start=12)

    # Add the road network graph to the folium map
    ox.plot_graph_folium(G, graph_map=map_osm, node_labels=True, edge_width=20)

    # Add markers for the source and destination nodes (source node is marked in green, destination node is marked in red)
    folium.Marker(location=(G.nodes[source_node]['y'], G.nodes[source_node]['x']), icon=folium.Icon(color='green'), popup=f'Source<br>Distance: {distance:.2f} meters').add_to(map_osm)


    for target_node in target_nodes:
      folium.Marker(location=(G.nodes[target_node]['y'], G.nodes[target_node]['x']), icon=folium.Icon(color='red'), popup='Destination').add_to(map_osm)

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

    # Get the coordinates of the shortest path
    shortest_path_coords = []
    for i in range(len(shortest_path)-1):
        edge = (shortest_path[i], shortest_path[i+1], 0)
        edge_coords = gdf_edges.loc[edge]['geometry']
        shortest_path_coords.extend([(point[1], point[0]) for point in edge_coords.coords])

    # Add the shortest path to the map as a PolyLine
    folium.PolyLine(locations=shortest_path_coords, color='blue', weight=5).add_to(map_osm)

    # Display the folium map inline
    display(map_osm)

# %% [markdown]
# ## Heuristic Functions
# 
# ### Euclidean Distance
# Euclidean distance, also known as straight-line distance or L2 distance, measures the straight-line distance between two points in Euclidean space. It is calculated using the formula:
# 
# $$
# \text{Euclidean Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
# $$
# where $(x_1, y_1)$ and $(x_2, y_2)$ are the coordinates of the two points.
# 
# ### Manhattan Distance
# Manhattan distance, also known as city block distance or L1 distance, measures the distance between two points in a grid-based system. It is calculated as the sum of the absolute differences of their coordinates:
# $$
# \text{Manhattan Distance} = |x_2 - x_1| + |y_2 - y_1|
# $$
# 
# ### Haversine Distance
# Haversine distance is used to calculate the distance between two points on the surface of a sphere, such as the Earth, given their longitude and latitude. It is calculated using the Haversine formula:
#  $$
# a = \sin^2\left(\frac{\Delta \text{lat}}{2}\right) + \cos(\text{lat}_1) \cdot \cos(\text{lat}_2) \cdot \sin^2\left(\frac{\Delta \text{lon}}{2}\right)
# $$
# 
# $$
# c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right)
# $$
# 
# $$
# \text{Haversine Distance} = R \cdot c
# $$
# 
# where `R = 6371.0` is the radius of the Earth, and $\Delta \text{lat}$ and $\Delta \text{lon}$ are the differences in latitude and longitude between the two points, respectively.
# 

# %% [markdown]
# #### Helpful code for calculating latitude, longitude, and node distances 
# 
# 

# %%
loc = "LUMS Lahore, Pakistan"
dist = 500

G = get_road_network(loc, dist)

# Print nodes information
for node, data in G.nodes(data=True):
    print(f"Node {node}: Latitude - {data['y']}, Longitude - {data['x']}")

# Print edges information
for u, v, data in G.edges(data=True):
    print(f"Edge ({u}, {v}): Length - {data['length']}")

# %% [markdown]
# ## <center> Task 0: Implement Heuristics </center>

# %%
def euclidean_heuristic(graph, node1, node2):
    x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
    x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
    
    dx = x2 - x1
    dy = y2 - y1
    
    euclidean_distance = (((dx * dx) + (dy * dy)) ** 0.5)
    return euclidean_distance

def manhattan_heuristic(graph, node1, node2):
    x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
    x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
    
    dx = x2 - x1
    dy = y2 - y1
    
    manhattan_distance = abs(dx) + abs(dy)
    return manhattan_distance

def haversine_heuristic(graph, node1, node2):

    earth_radius = 6371
    coordinates1 = graph.nodes[node1]
    coordinates2 = graph.nodes[node2]
    latitude1, longitude1 = radians(coordinates1['y']), radians(coordinates1['x'])
    latitude2, longitude2 = radians(coordinates2['y']), radians(coordinates2['x'])
    d_latitude = latitude2 - latitude1
    d_longitude = longitude2 - longitude1
    a = sin(d_latitude / 2) ** 2 + cos(latitude1) * cos(latitude2) * sin(d_longitude / 2) ** 2
    b = atan2(sqrt(a), sqrt(1 - a)) * 2
    haversine_distance = earth_radius * b
    return haversine_distance


# %% [markdown]
# ## <center> Task 1: A* Algorithm Implementation </center>
# 
# Implement the A* algorithm to find the shortest path between a source node and a target node. 
# 
# ### Helpful Pointers
# - You may use a priority queue to store nodes to be explored. Additionally, you may initialize dictionaries to keep track of the path and cost to reach each node from the start node.
# 
# - Once the goal node is reached, reconstruct the shortest path by backtracking through the dictionary, starting from the goal node and ending at the start node.
# 
# - Remember to track & return the total distance for use with the visualization functions later
# 
# 

# %%
def a_star(graph, start, goal, heuristic_func):
    p_queue = queue.PriorityQueue()
    p_queue.put((0, start, [start]))
    cost_nodes = {start: 0}

    while not p_queue.empty():
        current_cost, current, path = p_queue.get()

        if current == goal:
            return path 

        for next_node in graph.neighbors(current):
            new_cost = current_cost + graph[current][next_node][0]['length']
            if next_node not in cost_nodes or new_cost < cost_nodes[next_node]:
                cost_nodes[next_node] = new_cost
                priority = new_cost + heuristic_func(graph, next_node, goal)
                p_queue.put((priority, next_node, path + [next_node]))  

    return [] 

# %% [markdown]
# ## <center> Task 2: Best-First Search Algorithm Implementation </center>
# 
# Implement the Best-First Search algorithm to find a path between a source node and target node. Feel free to make helper functions if you want.
# 

# %%
def best_first_search(graph, start, goal, heuristic_func=euclidean_heuristic):
    p_queue = queue.PriorityQueue()
    p_queue.put((heuristic_func(graph, start, goal), start))
    
    came_from = {start: None}
    
    while not p_queue.empty():
        _, current = p_queue.get()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1]  

        for neighbor in graph.neighbors(current):
            if neighbor not in came_from: 
                came_from[neighbor] = current
                heuristic_cost = heuristic_func(graph, neighbor, goal)
                p_queue.put((heuristic_cost, neighbor))
    
    return None 

# %% [markdown]
# ## <center> Task 3: Informed Breadth-First Search Implementation </center>
# In this task, you'll implement a twist on the vanilla Breadth-First Search (BFS) algorithm. BFS is a blind search algorithm, thus it explores all possibilities without considering the cost, but integrating heuristic information can make it better (thus an "informed" search algorithm). <br/>
# 
# Use the Euclidean heuristic with BFS in the implementation below.
# 
# 
# 
# 
# 

# %%
def bfs(graph, start, goal, heuristic_func=euclidean_heuristic):
    visited = set()  
    queue = [(0, start, [start])]  
    while queue:
        queue.sort(key=lambda x: x[0])
        cost, current, path = queue.pop(0)
        
        if current == goal:
            return path 
        
        if current not in visited:
            visited.add(current)
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    estimated_cost = heuristic_func(graph, neighbor, goal)
                    new_path = path + [neighbor]
                    queue.append((estimated_cost, neighbor, new_path))

    return [] 

# %% [markdown]
# # Running the visualisations
# Now it's time to find out the shortcuts in LUMS! Write a `main` function that will allow you to visualise the different algorithms using the starter functions provided at the beginning of the assignment.
# 
# ### Steps
# 1. Build the initial graph network and visualise it once.
# 
# 3. Choose the source node as `810005319` (SSE) and target node as `11337034500` (SDSB). You can view these node values in the map by hovering too!
# 
# 4. Get the shortest route and visualise it for EACH algo you have implemented. For A star, use all three heuristics and visualise. Note that you will need to send your target node in a list to the `visualize_path_folium` function (check its arguments above).
# 
# Note: You can change the source and target for fun too, however we only expect you to visualize the paths from the provided source (SSE) and target (SDSB) destinations.

# %%
location = "LUMS Lahore, Pakistan"
distance = 500
source = 810005319 # SSE
destination = 11337034500 # SDSB

from IPython.display import display, HTML

def main():
    
    display(HTML('<b style="font-size: large;">Road Network: </b>'))
    R = get_road_network(location, distance)
    visualize_road_network(G, location, distance)

    # 1) a_star visualisation with euclidean
    display(HTML('<b style="font-size: large;">A* Algorithm with Euclidean Heuristic: </b>'))
    shortest_path_Astar_Euclidean = a_star(R, source, destination, euclidean_heuristic)
    visualize_path_folium(R, shortest_path_Astar_Euclidean, location, source, [destination], distance)
    
    # 2) a_star visualisation with manhattan
    display(HTML('<b style="font-size: large;">A* Algorithm with Manhattan Heuristic: </b>'))
    shortest_path_Astar_Manhattan = a_star(R, source, destination, manhattan_heuristic)
    visualize_path_folium(R, shortest_path_Astar_Manhattan, location, source, [destination], distance)
    
    # 3) a_star visualisation with haversine
    display(HTML('<b style="font-size: large;">A* Algorithm with Haversine Heuristic: </b>'))
    shortest_path_Astar_Haversine = a_star(R, source, destination, haversine_heuristic)
    visualize_path_folium(R, shortest_path_Astar_Haversine, location, source, [destination],distance)
    
    # 4) best_first visualisation
    display(HTML('<b style="font-size: large;">Best-First Search with Euclidean Heuristic: </b>'))
    shortest_path_BestFirst = best_first_search(R, source, destination, euclidean_heuristic)
    visualize_path_folium(R, shortest_path_BestFirst, location, source, [destination],distance)
    
    # 5) informed bfs visualisation
    display(HTML('<b style="font-size: large;">Informed BFS with Euclidean Heuristic: </b>'))
    shortest_path_InformedBFS = bfs(R, source, destination, euclidean_heuristic)
    visualize_path_folium(R, shortest_path_InformedBFS, location, source, [destination], distance)

if __name__ == "__main__":
    main()

# %% [markdown]
# # A-star Search using NetworkX
# 
# This function utilizes the built-in A* algorithm provided by the NetworkX library to find the shortest path between two nodes in a graph. Use this along with the visualisation functions provided earlier to compare your A_star implementation if you want!!
# 
# 

# %%
def astar_networkx_path(G, source, target):
  return nx.astar_path(G, source=source, target=target, weight='length')

shortest_path_Astar_NetworkX = astar_networkx_path(G, source, destination)
visualize_path_folium(G, shortest_path_Astar_NetworkX , location, source, [destination], distance)

# %% [markdown]
# ## <center> Task 4: Single Source And Multiple Destinations using A star </center>
# 

# %% [markdown]
# Now use your code from a_star implementation and modify it to go to multiple destinations in the most optimal way. Use the euclidean heuristic. Then visualize the shortest route obtained by your implementation using the`visualize_path_folium` function.

# %%
def a_star_multiple(graph, start, destinations, heuristic_func):
    if not destinations:
        return []
      
    route = []
    current_start = start
    remaining_destinations = set(destinations)

    while remaining_destinations:
        shortest_path = None
        min_distance = float('inf')
        for destination in remaining_destinations:
          path = a_star(graph, current_start, destination, heuristic_func)
          path_distance = 0
          for i in range(len(path) - 1):
              u, v = path[i], path[i + 1]
              path_distance = path_distance + graph[u][v][0]['length']
          
          if path_distance < min_distance:
              min_distance = path_distance
              shortest_path = path
              next_start = destination

        route.extend(shortest_path[:-1])
        current_start = next_start
        remaining_destinations.remove(current_start)

    route.append(current_start)

    return route

# %%
source = 810005319 # SSE

dest1 =  11336997534 #Cricket Ground
dest2 =  809970907 #Masjid
dest3 =  765049365 #law school

destinations = [dest1, dest2, dest3]

shortest_route = a_star_multiple(G, source, destinations, euclidean_heuristic)
visualize_path_folium(G, shortest_route, location, source, destinations, distance)

# %% [markdown]
# # Analysis
# Identify the algortihm which provided you with the shortest path. Compare it with the other graphs and explain why it was the best.
# 
# All the algorithms (i.e. three variants of A*, Best-first and Informed Breadth-first) found the same shortest path between the source (SSE) and the destination (SDSB). However, this was not the case when the source and destination were changed or they were further away from each other. In such cases, informed BFS and Best-first Search would fail to yield the optimal shortest path. 
# The A* algorithm was the best among all for the following reasons: <br>
# 1. It always found the optimal shortest path between ANY source and destination <br>
# 2. It used the haversine heuristic which provides highly accurate results in calculating real physical distances. Thus, the A* algorithm using haversine heuristic can be used in finding optimal paths over larger distances.


