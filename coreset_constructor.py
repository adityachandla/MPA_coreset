from data import RepresentativePoint, Square
from plotter import Plotter
from sklearn.cluster import KMeans
import math


class CoresetConstructor:
    _instance_id = 1

    def __init__(self, points, epsilon=0.5, k=4, d=2):
        self.points = points
        self.epsilon = epsilon
        self.k = k
        self.d = d
        self.representative_points = list()
        self.plotter = Plotter("./images/", CoresetConstructor._instance_id)
        CoresetConstructor._instance_id += 1

    def get_k_means_centers(self) -> list[list[float]]:
        k_means = KMeans(n_clusters=self.k, n_init='auto').fit([p.coords for p in self.points])
        return k_means.cluster_centers_

    @staticmethod
    def dist_square(p1: [float,float], p2: [float,float]) -> float:
        return (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2

    @staticmethod
    def dist(p1: [float,float], p2: [float,float]) -> float:
        return math.sqrt(CoresetConstructor.dist_square(p1, p2))

    def compute_cost(self, centers) -> float:
        total_cost = 0
        for point in self.points:
            costs = []
            for center in centers:
                costs.append(self.dist_square(point.coords, center))
            total_cost += min(costs)
        return total_cost

    def get_points_in_annulus(self, center, radius):
        points_inside = []
        points_remaining = []
        for point in self.points:
            distance = CoresetConstructor.dist(center, point.coords)
            if distance <= radius:
                points_inside.append(point)
            else:
                points_remaining.append(point)
        self.points = points_remaining
        return points_inside

    def get_num_squares(self, center, radius) -> int:
        side_length = (self.epsilon*radius)/math.sqrt(2)
        return math.ceil((2*radius)/side_length)

    def add_representatives(self, center, prev_radius, radius):
        points_in_annulus = self.get_points_in_annulus(center, radius)
        side_length = (self.epsilon*radius)/math.sqrt(2)
        num_squares = math.ceil((2*radius)/side_length)
        point_counters = [[0 for i in range(num_squares) ] for j in range(num_squares)]
        x_min = center[0]-radius ## left side
        y_min = center[1]-radius ## bottom side
        # print(f"Circle bottom left ({x_min}, {y_min})")
        # print(f"Side length is {side_length}");
        for point in points_in_annulus:
            x_offset = int(math.floor((point.coords[0]-x_min)/side_length)) ## How many tiles right
            y_offset = int(math.floor((point.coords[1]-y_min)/side_length)) ## How many tiles up
            if x_offset > num_squares or y_offset > num_squares or x_offset < 0 or y_offset < 0:
                print(f"{x_offset} {y_offset} point={point}")
            point_counters[x_offset][y_offset] += point.weight
        ## Add representative points
        start_x = center[0] - radius
        start_y = center[1] - radius 
        for y in range(num_squares):
            for x in range(num_squares):
                center_x = start_x + (side_length/2)
                center_y = start_y + (side_length/2)
                if point_counters[x][y] > 0:
                    p = RepresentativePoint(coords=(center_x, center_y), weight=point_counters[x][y])
                    self.representative_points.append(p)
                start_x += side_length
            start_y += side_length
            start_x = center[0]-radius
        
    def get_coreset(self) -> list[RepresentativePoint]:
        n = len(self.points)
        self.plotter.plot_representative_points(self.points)
        self.plotter.save()
        self.plotter.clear()
        centers = self.get_k_means_centers()
        cost_original = self.compute_cost(centers)
        radius = math.sqrt(cost_original/(n*math.log(n)))
        prev_radius = 0
        while len(self.points) > 0:
            for center in centers:
                self.add_representatives(center, prev_radius, radius)
                # Plotting annulus
                self.plotter.plot_circle(center, radius)
                if prev_radius > 0:
                    self.plotter.plot_circle(center, prev_radius)
            ## Plot of all the representative points
            self.plotter.plot_representative_points(self.representative_points)
            self.plotter.save()
            self.plotter.clear()
            prev_radius = radius
            radius *= 2
        return self.representative_points
