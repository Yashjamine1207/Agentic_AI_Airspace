"""
core/astar.py
3D A* search on the 100×100×20 TRACON grid.
26-connected neighbourhood, Manhattan-distance heuristic.
"""

import heapq
import numpy as np

TRACON = {
    "lat_min": 36.5, "lat_max": 38.5,
    "lon_min": -123.0, "lon_max": -121.0,
    "alt_min": 0.0,   "alt_max": 18000.0,
    "grid_x": 100,    "grid_y": 100, "grid_z": 20,
}


def latlon_to_grid(lat, lon, alt):
    x = int((lat - TRACON["lat_min"]) / (TRACON["lat_max"] - TRACON["lat_min"]) * 99)
    y = int((lon - TRACON["lon_min"]) / (TRACON["lon_max"] - TRACON["lon_min"]) * 99)
    z = int((alt - TRACON["alt_min"]) / (TRACON["alt_max"] - TRACON["alt_min"]) * 19)
    return (np.clip(x, 0, 99), np.clip(y, 0, 99), np.clip(z, 0, 19))


def grid_to_latlon(x, y, z):
    lat = TRACON["lat_min"] + x / 99 * (TRACON["lat_max"] - TRACON["lat_min"])
    lon = TRACON["lon_min"] + y / 99 * (TRACON["lon_max"] - TRACON["lon_min"])
    alt = TRACON["alt_min"] + z / 19 * (TRACON["alt_max"] - TRACON["alt_min"])
    return lat, lon, alt


def astar_3d(start_grid, goal_grid, forbidden_zones=None):
    """
    A* search on 3D grid.
    Returns path as list of (x, y, z) tuples.
    """
    forbidden = set(forbidden_zones or [])

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

    neighbors = [
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    open_set  = [(0, start_grid)]
    came_from = {}
    g_score   = {start_grid: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_grid:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_grid)
            return path[::-1]

        for dx, dy, dz in neighbors:
            nb = (current[0]+dx, current[1]+dy, current[2]+dz)
            if not (0 <= nb[0] < 100 and 0 <= nb[1] < 100 and 0 <= nb[2] < 20):
                continue
            if nb in forbidden:
                continue
            tentative_g = g_score[current] + np.sqrt(dx**2 + dy**2 + dz**2)
            if tentative_g < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb]   = tentative_g
                f = tentative_g + heuristic(nb, goal_grid)
                heapq.heappush(open_set, (f, nb))

    return []  # no path


def constraint_to_forbidden(constraint, grid_size=(100, 100, 20)):
    """Convert NOTAM JSON constraint → set of forbidden grid cells."""
    forbidden = set()
    if constraint.get("latitude_center") is None:
        return forbidden
    c_lat   = constraint["latitude_center"]
    c_lon   = constraint["longitude_center"]
    radius  = constraint["radius_nm"] / 60.0
    floor   = constraint["altitude_floor_ft"]
    ceiling = constraint["altitude_ceiling_ft"]

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            lat = TRACON["lat_min"] + x / 99 * (TRACON["lat_max"] - TRACON["lat_min"])
            lon = TRACON["lon_min"] + y / 99 * (TRACON["lon_max"] - TRACON["lon_min"])
            dist = np.sqrt((lat - c_lat)**2 + (lon - c_lon)**2)
            if dist < radius:
                z_min = int((floor - TRACON["alt_min"]) /
                            (TRACON["alt_max"] - TRACON["alt_min"]) * 19)
                z_max = int((ceiling - TRACON["alt_min"]) /
                            (TRACON["alt_max"] - TRACON["alt_min"]) * 19)
                for z in range(max(0, z_min), min(20, z_max + 1)):
                    forbidden.add((x, y, z))
    return forbidden


def path_to_latlon(path):
    """Convert grid path to (lat, lon, alt) coordinates."""
    return [grid_to_latlon(x, y, z) for x, y, z in path]
