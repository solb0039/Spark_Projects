# Name: Sean Solberg
# Decription: Example of k-means clustering algorithm using Spark
# Assumes input files are in current directory
# Usage: cluster.py
# Output: 2 files generated with 20 iterations of Euclidean distances for input files of clusters (c1 and c2)

from pyspark import SparkConf, SparkContext
import numpy as np
conf = SparkConf()
sc = SparkContext(conf=conf)

# FUNCTION TO READ DATA AND STORE AS NUMPY ARRAY

def load_data(filename):
    return np.genfromtxt(filename, delimiter="")

# Assume files are located in current directory
c1 = np.array(load_data("./c1.txt"))
c2 = np.array(load_data("./c2.txt"))

# FUNCTION TO CALCULATE MEAN OF ARRAY OF ARRAYS
def calc_mean(item):
    return np.divide(item[0], item[1])

# READ DATA
data = sc.textFile("../HW2/data.txt")
data2 = data.map(lambda x: x.encode('ascii', 'ignore'))
# MAP STRING DATA TO FLOAT
data3 = data2.map(lambda line: np.array([float(x) for x in line.split(' ')]))

# FUNCTION THAT CALCS L2 NORM DISTANCE AND RETURNS CLUSTER ID AND ERROR
def dist_test(point, c1, order):
    cluster = 0
    clus_count = 0
    min_dist = float("inf")
    for clus in c1:
        clus_count += 1
        # ord =2 for Eculidean and 1 for Manhattan
	dist = np.linalg.norm(point - clus, ord=order)
        if dist <= min_dist:
            min_dist = dist
            cluster = clus_count
    return(cluster, min_dist**1.)

# ITERATIVE LOOP OVER COST FUNCTION
error = []
def calc_cluster(file, ord):
    MAX_ITER = 21
    for loop in range(1,MAX_ITER):
        print("Iteration: ", loop)
        print("(Cluster, Num Points)")

        # MAP TO CLUSTER AS ID AND COORDINATES AS VALUE
        data4_t = data3.map(lambda x:  (dist_test(x, file, ord), x) )
        
	#SPLIT DATA INTO TWO WITH (CLUSTER,ERROR) AND (CLUSTER, X DATA)
        test4 = data4_t.map(lambda x: (x[0][0],  (x[1])))
        test4c = data4_t.map(lambda x: (x[0][0],  (x[0][1])) )
        
	# GET LENGTHS AND COSTS
        len_test = test4.groupByKey().map(lambda x: (x[0], len(x[1]))).sortByKey()
        cost = test4c.groupByKey().map(lambda x: (1, sum(x[1]))).reduceByKey(lambda x,y: x+y)
        print(len_test.take(10))
        print("Error is: ")
        print(cost).take(1)
        error.append(cost.map(lambda x: (x[1])).take(1))
        
	# SUM THE X ARRAYS
	data5_test = test4.reduceByKey(lambda x,y: x+y )
        
	# JOIN THE SUM ARRAY AND THE LENGTHS TO CALCULATE THE MEAN OF ALL DATA POINTS ASSIGNED TO CLUSTER
	data6_test = data5_test.union(len_test).groupByKey().map(lambda x: (x[0], list(x[1]))).sortByKey()
        data7_test = data6_test.map(lambda pair: calc_mean(pair[1]))
        
	# RE-ASSIGN TO CLUSTER ARRAYS TO UPDATE THEIR CENTERS
        file =[x for x in data7_test.toLocalIterator()]

print(error)

# OUTPUT
calc_cluster(c1, 2)  #Calc. Euclidean distance with c1
sc.parallelize(error).saveAsTextFile("c1_error_Euclidean")
error = []
calc_cluster(c2, 2)  #Calc. Euclidean distance with c2
sc.parallelize(error).saveAsTextFile("c2_error_Euclidean")
