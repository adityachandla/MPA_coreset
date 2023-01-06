import math
import os
from time import time
import random
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
from sklearn.datasets import make_blobs
from coreset_constructor import CoresetConstructor
from data import RepresentativePoint
from pyspark import SparkConf, SparkContext
from plotter import Plotter
from argparse import ArgumentParser

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

def main():
    parser = ArgumentParser()
    parser.add_argument("--output", help="Output directory for images (default=./images)", type=str, default="./images/")
    parser.add_argument("--epsilon", help="Epsilon for the algorithm (default=0.7)", type=float, default=0.7)
    parser.add_argument("--machines", help="Number of machines in the first round (default=4)", type=int, default=4)
    parser.add_argument("--clusters", help="Number of clusters in generated data (default=4)", type=int, default=4)
    parser.add_argument("--points", help="Number of points in generated data (default=1000)", type=int, default=1000)

    args = parser.parse_args()
    create_subdirectory(args)
    blob_points, cluster = make_blobs(n_samples=args.points, centers=args.clusters)
    representative_points = [RepresentativePoint(coords=p, weight=1) for p in blob_points]
    coreset = get_coreset_distributed(args, representative_points)

if __name__ == "__main__":
    main()
