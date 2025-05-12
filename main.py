import numpy as np
from shapely import unary_union
from shapely.geometry import Polygon, LineString
from shapely.affinity import translate, rotate
from shapely.ops import polygonize
import matplotlib.pyplot as plt
import random
import math

seed = 45
np.random.seed(seed)
random.seed(seed)

def generate_concave_polygon(n_points=6):
    assert n_points >= 4, "Concave polygon needs at least 4 points"
    
    while True:
        angles = np.sort(np.random.rand(n_points) * 2 * np.pi)
        radii = np.random.uniform(10, 20, size=n_points)
        points = np.c_[np.cos(angles), np.sin(angles)] * radii[:, None]

        idx = np.random.randint(0, n_points)
        points[idx] *= 0.4  # pull it toward the center

        polygon = Polygon(points)
        polygon = translate(polygon, 5 + random.random() * 20, 5 + random.random() * 20)

        if polygon.is_valid and polygon.is_simple and not polygon.is_empty:
            return polygon

def generate_arc_points(start_point, angle_deg, origin, num_segments=30):
    """Generates points approximating a circular arc."""
    if abs(angle_deg) < 1e-6: return [start_point]
    start_vec = np.array(start_point) - np.array(origin)
    radius = np.linalg.norm(start_vec)
    if radius < 1e-9: return [start_point]
    start_angle_rad = math.atan2(start_vec[1], start_vec[0])
    end_angle_rad = start_angle_rad + math.radians(angle_deg)
    angles = np.linspace(start_angle_rad, end_angle_rad, num_segments + 1)
    points = [(origin[0] + radius * math.cos(a), origin[1] + radius * math.sin(a)) for a in angles]
    return points


def find_closest_point_on_segment(p1, p2, origin=(0, 0)):
    """Finds the point on the line segment [p1, p2] closest to the origin."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    orig = np.array(origin)

    vec_p1p2 = p2 - p1
    len_sq = np.dot(vec_p1p2, vec_p1p2)

    if len_sq < 1e-12: # p1 and p2 are virtually the same point
        return tuple(p1), 0.0 # Return p1 and parameter t=0

    # Project origin onto the line containing the segment
    # Parameter t = dot(Origin - P1, P2 - P1) / |P2 - P1|^2
    vec_p1orig = orig - p1
    t = np.dot(vec_p1orig, vec_p1p2) / len_sq

    if t < 0.0:
        # Closest point is p1
        return tuple(p1), 0.0
    elif t > 1.0:
        # Closest point is p2
        return tuple(p2), 1.0
    else:
        # Closest point is the projection, which lies on the segment
        projection = p1 + t * vec_p1p2
        return tuple(projection), t


def get_boundary_lines(p_start, p_end, total_rotation_angle_deg, origin=(0, 0), arc_segments: int = 30):
    boundary_lines = [p_start.exterior, p_end.exterior]

    vertices = list(p_start.exterior.coords)[:-1]
    for i, v_start in enumerate(vertices):
        arc_pts = generate_arc_points(v_start, total_rotation_angle_deg, origin, arc_segments)
        if len(arc_pts) >= 2:
            # Ensure the arc LineString is valid
            arc_line = LineString(arc_pts)
            if arc_line.is_valid and not arc_line.is_empty:
                    boundary_lines.append(arc_line)

    edges_processed = 0
    edge_min_arcs_added = 0

    epsilon = 1e-6
    coords = list(p_start.exterior.coords)
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i+1]
        edges_processed += 1

        p_min, t = find_closest_point_on_segment(p1, p2, origin)

        # Check if the closest point is strictly between the endpoints
        if epsilon < t < 1.0 - epsilon:
            # This point p_min traces an important inner boundary arc
            arc_pts = generate_arc_points(p_min, total_rotation_angle_deg, origin, arc_segments)
            if len(arc_pts) >= 2:
                arc_line = LineString(arc_pts)
                if arc_line.is_valid and not arc_line.is_empty:
                    boundary_lines.append(arc_line)
                    edge_min_arcs_added += 1
                else: print(f"Warning: Invalid edge min arc for edge {i}") # Debugging

    return boundary_lines


total_rotation_angle_deg = 55
arc_segments = 10
origin = (0, 0)

p_start = generate_concave_polygon()
p_end = rotate(p_start, total_rotation_angle_deg, origin)

boundary_lines = get_boundary_lines(p_start, p_end, total_rotation_angle_deg, origin, 10)
polygons_from_lines = list(polygonize(unary_union(boundary_lines)))
swept_polygon = unary_union(polygons_from_lines)

def plot_poly(p):
    x, y = p.exterior.xy
    plt.plot(x, y, 'o-', label='Concave Polygon')
    plt.fill(x, y, alpha=0.4)


def plot_lines(lines):
    for line in lines:
        x, y = line.xy
        plt.plot(x, y, color='black', linewidth=3)


plt.figure(figsize=(12, 12))
plt.plot(0, 0, 'o', markersize=10, color='black')

polys = [rotate(p_start, rot, origin) for rot in [0, 20, 40, 55]]
for p in polys:
    plot_poly(p)
plot_lines(boundary_lines)

plt.gca().set_aspect('equal')
plt.legend()
plt.savefig('out.png')

plt.close()

plt.figure(figsize=(12, 12))
plt.plot(0, 0, 'o', markersize=10, color='black')

plot_poly(swept_polygon)
plot_lines(boundary_lines)

plt.gca().set_aspect('equal')
plt.legend()
plt.savefig('sweep.png')

plt.close()
