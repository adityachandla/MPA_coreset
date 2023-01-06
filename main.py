import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
from sklearn.datasets import make_blobs
from coreset_constructor import CoresetConstructor
from data import RepresentativePoint
from pyspark import SparkConf, SparkContext

num_clusters = 4
num_points = 10000

def plot(dataset):
    plt.scatter(dataset[0][:,0], dataset[0][:,1])
    plt.savefig("./blobs.png")

def partition_points(points, num_partitions):
    random.shuffle(points)
    partitions = [[] for i in range(num_partitions)]
    for i in range(len(points)):
        partitions[(i%num_partitions)].append(points[i])
    return partitions

def construct_coreset(rep_points):
    coresetConstructor = CoresetConstructor(rep_points, epsilon=0.8, k=num_clusters)
    return coresetConstructor.get_coreset()

def main():
    blob_points, cluster = make_blobs(n_samples=num_points, centers=num_clusters)
    representative_points = [RepresentativePoint(coords=p, weight=1) for p in blob_points]

    num_machines = math.floor(math.sqrt(num_points))
    partitions = partition_points(representative_points, num_machines)

    conf = SparkConf().setAppName("Coreset Construction").setMaster("local")
    sc = SparkContext(conf=conf)
    iteration = 1
    while len(partitions) > 1:
        rep_partitions = sc.parallelize(partitions).map(lambda partition : construct_coreset(partition)).collect()
        ## Join two partitions together
        for i in range(1, len(rep_partitions), 2):
            rep_partitions[i-1].extend(rep_partitions[i])
        partitions = rep_partitions[::2]
        print(f"Finished iteration {iteration}")
        iteration += 1
    final_points = partitions[0]
    print(f"Got {len(final_points)} representative points")

if __name__ == "__main__":
    main()
