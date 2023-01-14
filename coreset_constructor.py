from data import RepresentativePoint, Square
from sklearn.cluster import KMeans
import math
import dist


class CoresetConstructor:
    _instance_id = 1

    def __init__(self, points, args):
        self.points = points
        self.epsilon = args.epsilon
        self.k = args.clusters
        self.d = 2
        self.representative_points = list()
        CoresetConstructor._instance_id += 1

    def get_k_means_centers(self) -> list[list[float]]:
        k_means = KMeans(n_clusters=self.k, n_init='auto').fit([p.coords for p in self.points])
        return k_means.cluster_centers_

    def compute_cost(self, centers) -> float:
        total_cost = 0
        for point in self.points:
            costs = []
            for center in centers:
                costs.append(dist.dist_square(point.coords, center))
            total_cost += min(costs)
        return total_cost

    def get_points_in_annulus(self, center, radius):
        points_inside = []
        points_remaining = []
        for point in self.points:
            distance = dist.dist(center, point.coords)
            if distance <= radius:
                points_inside.append(point)
            else:
                points_remaining.append(point)
        self.points = points_remaining
        return points_inside

    def get_num_squares(self, center, radius) -> int:
        side_length = (self.epsilon*radius)/math.sqrt(2)
        return math.ceil((2*radius)/side_length)

    def add_representatives(self, center, radius):
        points_in_annulus = self.get_points_in_annulus(center, radius)
        side_length = (self.epsilon*radius)/math.sqrt(2)
        num_squares = math.ceil((2*radius)/side_length)
        point_counters = [[0 for i in range(num_squares) ] for j in range(num_squares)]
        x_min = center[0]-radius 
        y_min = center[1]-radius
        for point in points_in_annulus:
            x_offset = int(math.floor((point.coords[0]-x_min)/side_length))
            y_offset = int(math.floor((point.coords[1]-y_min)/side_length))
            assert x_offset <= num_squares and x_offset >= 0
            assert y_offset <= num_squares and y_offset >= 0
            point_counters[x_offset][y_offset] += point.weight
        self.add_representatives_for_grids(point_counters, center, radius, side_length)

    def add_representatives_for_grids(self, counters, center, radius, side_length):
        start_x = center[0] - radius
        start_y = center[1] - radius 
        for y in range(len(counters)):
            for x in range(len(counters[0])):
                center_x = start_x + (side_length/2)
                center_y = start_y + (side_length/2)
                if counters[x][y] > 0:
                    p = RepresentativePoint(coords=(center_x, center_y), weight=counters[x][y])
                    self.representative_points.append(p)
                start_x += side_length
            start_y += side_length
            start_x = center[0]-radius

    def get_coreset(self) -> list[RepresentativePoint]:
        n = len(self.points)
        centers = self.get_k_means_centers()
        cost_original = self.compute_cost(centers)
        radius = math.sqrt(cost_original/(n*math.log(n)))
        prev_radius = 0
        while len(self.points) > 0:
            for center in centers:
                self.add_representatives(center, radius)
            prev_radius = radius
            radius *= 2
        return self.representative_points
