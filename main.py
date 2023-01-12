import os
import dist
from time import time
import random
import math
from sklearn.datasets import make_blobs
from coreset_constructor import CoresetConstructor
from data import RepresentativePoint
from pyspark import SparkConf, SparkContext
from plotter import Plotter
from argparse import ArgumentParser
from sklearn.cluster import KMeans

def partition_points(points, num_partitions):
    random.shuffle(points)
    partitions = [[] for i in range(num_partitions)]
    for i in range(len(points)):
        partitions[(i%num_partitions)].append(points[i])
    return partitions

def construct_coreset(rep_points, args):
    coresetConstructor = CoresetConstructor(rep_points, args)
    return coresetConstructor.get_coreset()

def get_coreset_distributed(args, representative_points):
    partitions = partition_points(representative_points, args.machines)

    conf = SparkConf().setAppName("Coreset Construction").setMaster("local")
    sc = SparkContext(conf=conf)
    iteration = 1
    while len(partitions) > 1:
        rep_partitions = sc.parallelize(partitions).map(lambda partition : construct_coreset(partition, args)).collect()
        for i in range(1, len(rep_partitions), 2):
            rep_partitions[i-1].extend(rep_partitions[i])
        partitions = rep_partitions[::2]
        print(f"Finished iteration {iteration}")
        iteration += 1
    final_points = partitions[0]
    print(f"Got {len(final_points)} representative points")
    return final_points

def create_subdirectory(args):
    base = args.output
    if not os.path.exists(base):
        os.mkdir(base)
    epoch = str(round(time()*1000))
    if not base.endswith("/"):
        base += "/"
    base += f"run_{epoch}/"
    os.mkdir(base)
    args.output = base

def coreset_k_means(args, coreset) -> list[tuple[float,float]]:
    points = [p.coords for p in coreset]
    weights = [p.weight for p in coreset]
    k_means = KMeans(n_clusters=args.clusters, n_init='auto').fit(points, sample_weight=weights)
    return k_means.cluster_centers_

def initial_k_means(args, points) -> list[tuple[float,float]]:
    k_means = KMeans(n_clusters=args.clusters, n_init='auto').fit(points)
    return k_means.cluster_centers_

def compute_cost(points: list[tuple[float,float]], centers: list[tuple[float,float]]) -> float:
    total_cost = 0
    for point in points:
        min_cost = math.inf
        for center in centers:
            cost = dist.dist_square(point, center)
            if cost < min_cost:
                min_cost = cost
        total_cost += min_cost
    return total_cost

def main():
    parser = ArgumentParser()
    parser.add_argument("--output", help="Output directory for images (default=./images)", type=str, default="./images/")
    parser.add_argument("--epsilon", help="Epsilon for the algorithm (default=0.7)", type=float, default=0.7)
    parser.add_argument("--machines", help="Number of machines in the first round (default=4)", type=int, default=4)
    parser.add_argument("--clusters", help="Number of clusters in generated data (default=4)", type=int, default=4)
    parser.add_argument("--points", help="Number of points in generated data (default=1000)", type=int, default=1000)

    args = parser.parse_args()
    create_subdirectory(args)
    args.center_limit = [-100,100]
    blob_points, cluster = make_blobs(n_samples=args.points, centers=args.clusters, center_box=args.center_limit, cluster_std=10)
    result_file = open(args.output + "result.txt", "w+")

    normal_centers = initial_k_means(args, blob_points)
    Plotter.plot_initial_clusters(args, blob_points, cluster)

    representative_points = [RepresentativePoint(coords=p, weight=1) for p in blob_points]
    time_before = time()
    coreset = get_coreset_distributed(args, representative_points)
    time_after = time()
    result_file.write(f"epsilon={args.epsilon} machines={args.machines} clusters={args.clusters} points={args.points}")
    result_file.write(f"Total time taken={time_after-time_before}s\n")

    Plotter.plot_final(args, coreset)
    coreset_centers = coreset_k_means(args, coreset)
    coreset_cost = compute_cost(blob_points, coreset_centers)
    normal_cost = compute_cost(blob_points, normal_centers)
    result_file.write(f"Got coreset centers cost: {coreset_cost}\n")
    result_file.write(f"Cost of initial k-means++: {normal_cost}\n")
    result_file.close()

if __name__ == "__main__":
    main()
