from pyspark import SparkConf, SparkContext

def main():
    conf = SparkConf().setAppName("Coreset Construction").setMaster("local")
    sc = SparkContext(conf=conf)
    ## In coreset construction, we need to distribute a set of points to all machines
    data = [[(1,2,22), (2,3,44)], [(4,5,22), (11,12, 5)]]
    rdd = sc.parallelize(data, 2) # Here we can specify the number of machines/cores
    reduced_data = rdd.map(lambda x : x[0]).collect()
    print(reduced_data)
    print(type(reduced_data))

if __name__ == "__main__":
    main()
