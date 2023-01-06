import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
from sklearn.datasets import make_blobs
from coreset_constructor import CoresetConstructor
from data import RepresentativePoint
from pyspark import SparkConf, SparkContext
from plotter import Plotter

num_clusters = 4
num_points = 10000

def partition_points(points, num_partitions):
    random.shuffle(points)
    partitions = [[] for i in range(num_partitions)]
    for i in range(len(points)):
        partitions[(i%num_partitions)].append(points[i])
    return partitions

def construct_coreset(rep_points):
    coresetConstructor = CoresetConstructor(rep_points, epsilon=0.8, k=num_clusters)
    return coresetConstructor.get_coreset()

def get_coreset_distributed(representative_points):
    num_machines = 4
    partitions = partition_points(representative_points, num_machines)

    conf = SparkConf().setAppName("Coreset Construction").setMaster("local")
    sc = SparkContext(conf=conf)
    iteration = 1
    while len(partitions) > 1:
        rep_partitions = sc.parallelize(partitions).map(lambda partition : construct_coreset(partition)).collect()
        for i in range(1, len(rep_partitions), 2):
            rep_partitions[i-1].extend(rep_partitions[i])
        partitions = rep_partitions[::2]
        print(f"Finished iteration {iteration}")
        iteration += 1
    final_points = partitions[0]
    print(f"Got {len(final_points)} representative points")

def main():
    blob_points, cluster = make_blobs(n_samples=num_points, centers=num_clusters)
    representative_points = [RepresentativePoint(coords=p, weight=1) for p in blob_points]
    coreset = get_coreset_distributed(representative_points)

if __name__ == "__main__":
    main()
