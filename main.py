import math
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
from typing import NamedTuple
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

class RepresentativePoint(NamedTuple):
    coords: tuple[float,float]
    weight: int

class Square(NamedTuple): 
    topLeft: tuple[float,float]
    bottomRight: tuple[float]

def get_blobs():
    blobs = make_blobs(n_samples=100, centers=4)
    return blobs

class Coreset:
    def __init__(self, points, epsilon=0.5, k=4, d=2):
        self.points = points
        self.epsilon = epsilon
        self.k = k
        self.representative_points = list()

    def get_k_means_centers(self) -> list[list[float]]:
        k_means = KMeans(n_clusters=4, n_init='auto').fit(self.points)
        return k_means.cluster_centers_

    @staticmethod
    def dist_square(p1: [float,float], p2: [float,float]) -> float:
        return (p2[0]-p1[0])**2 + (p2[1]-p2[0])**2

    @staticmethod
    def dist(p1: [float,float], p2: [float,float]) -> float:
        return math.sqrt(dist_square(p1, p2))

    def compute_cost(self, centers) -> float:
        total_cost = 0
        for point in self.points:
            costs = []
            for center in centers:
                costs.append(dist_square(point, center))
            total_cost += min(costs)
        return total_cost

    def get_points_in_annulus(self, center, prev_radius, radius):
        point_inside = []
        points_remaining = []
        for point in self.points:
            distance = dist(center, point)
            if distance > prev_radius and distance <= radius:
                points_inside.append(point)
            else:
                points_remaining.append(point)
        self.points = points_remaining
        return points_inside

    def get_squares(self, center, radius) -> defaultdict[Square, int]:
        side_length = 

    def add_representatives(self, center, prev_radius, radius):
        points_in_annulus = self.get_points_in_annulus(center, prev_radius, radius)
        squares = self.get_squares(center, radius)
        

    def get_coreset(self) -> list[RepresentativePoint]:
        n = len(self.points)
        centers = self.get_k_means_centers()
        cost_original = self.compute_cost(centers)
        radius = math.sqrt(cost_original/(n*math.log(n)))
        prev_radius = 0
        while len(self.respresentative_points) < n:
            for center in centers:
                self.add_representatives(self, prev_radius, radius)
            prev_radius = radius
            radius *= 2
        return self.representative_points
    

def plot(dataset):
    plt.scatter(dataset[0][:,0], dataset[0][:,1])
    plt.savefig("./blobs.png")

def main():
    blob_points, cluster = get_blobs()
    get_coreset(blob_points, epsilon=0.8)
    # plot(blobs)

if __name__ == "__main__":
    main()
