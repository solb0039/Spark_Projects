# Name: Sean So:wqlberg
# Description: Example of Page Rank algorithm in Spark
# Input: File 'page_rank_points' containing graph connections. Assumes file is in current directory.
# Date: 2/22/2017

import os
import sys

# Point to Spark directory if not installed in a virtual environment
os.environ['SPARK_HOME'] = "/Users/Sean/spark"
os.environ['HADOOP_HOME'] = "/Users/Sean/spark"
sys.path.append("/Users/Sean/spark/python")
sys.path.append("/Users/Sean/spark/python/lib")

from pyspark import SparkConf, SparkContext
import numpy as np
conf = SparkConf()
import re
sc = SparkContext(conf=conf)

print("Part A, PageRank....")

#Read data into RDD
data = sc.textFile("page_rank_points.txt")
data2 = data.map(lambda line: ([int(x) for x in line.split('\t')]))

# PART A: PAGERANK ALGORITHM

#N, and intialize with vector weights 1/N
#n is 100 or 1000
nodes = 1000
R = np.full(nodes, 1./nodes)

beta = 0.8
constant = (1.-beta) / nodes
ITER = 40

# Function to multiply values by corresponding key in R vector times beta and add constant
def calc_R(key , value ):
    return beta*value*R[key-1]

#Degree of nodes: Group by key and then eliminate redundant entries by using list
dataReduced = data2.groupByKey().map(lambda x: (int(x[0]), list(sorted(set(x[1])))))

#Assign new value as 1/# nodes
node_degree = dataReduced.map(lambda x: (int(x[0]), 1./len(x[1])))

# Save data with redundant nodes removed
new_data = dataReduced.flatMapValues(lambda x: x)

# Matrix multiplication
data3 = new_data.join(node_degree).map(lambda (x,(y,z)): ((int(x),y),z))
for i in range(1,ITER):
    data4 = data3.map(lambda ((x,y),z): (y, calc_R(x, z)) ).reduceByKey(lambda x1, x2: x1+x2)
    file = [x for x in data4.toLocalIterator()]
    # Update R
    for key, val in file:
        R[int(key)-1] = (round(val, 4) + constant)

#Put R array into sorted array and print top values
Rs = []
for i in range(0,len(R)):
    Rs.append( (i+1, R[i]) )

# Print results
Rs = sorted(Rs, key= lambda x: -x[1])
print("Part A: PageRank")
print("Top 5 ids and PageRank: ", Rs[0:5])
print("Bottom 5 ids and PageRank: ", Rs[len(Rs)-6:len(Rs)-1])


##PART B: HITS ALGORITHM

print("Part B, HITS....")

#Get rid of duplicate entries in data
dR = data2.groupByKey().map(lambda x: (int(x[0]), list(sorted(set(x[1])))))
dR2 = dR.flatMapValues(lambda x: x)

nodes = 1000
h = np.full(nodes, 1.)
a = np.full(nodes, 0.)

#Build L transpose
L_T = dR2.map(lambda x:  ((x[0],x[1]), 1) )

#Build L
L = dR2.map(lambda x: ((x[1],x[0]),1))

def calc_a(key , value ):
    return value*h[key-1]

def calc_h(key , value ):
    return value*a[key-1]

ITER = 40
for i in range(1,ITER):
    #Compute a by multiplying Lt with h
    temp_a = L_T.map(lambda ((k1, k2), v): (k2, calc_a(k1, v))).reduceByKey(lambda x1, x2: x1 + x2)
    #Write a
    file_a = [x for x in temp_a.toLocalIterator()]
    # Update R
    for key, val in file_a:
        a[int(key)-1] = val
    #Scale a to 1
    a = a / a.max()
    #Compute  h = La
    temp_h = L.map(lambda ((k1, k2), v): (k2, calc_h(k1, v))).reduceByKey(lambda x1, x2: x1 + x2)
    file_h = [x for x in temp_h.toLocalIterator()]
    # Update R
    for key, val in file_h:
        h[int(key)-1] = val
    #Scale a to 1
    h = h / h.max()


h_data = []
a_data = []

for i in range(0,len(a)):
    a_data.append( (i+1, a[i]) )

for i in range(0,len(h)):
    h_data.append( (i+1, h[i]) )

# Print results
a_data = sorted(a_data, key= lambda x: -x[1])
h_data = sorted(h_data, key= lambda x: -x[1])
print("Part B: HITS")
print("Hubbiness")
print("Top 5 ids with hubbiness score: ", h_data[0:5])
print("Lowest 5 ids and hubbiness score: ", h_data[ (len(h_data)-5) : (len(h_data)) ])
print("Authority")
print("Top 5 ids with authority score: ", a_data[0:5])
print("Lowest 5 ids and authority score: ", a_data[len(a)-5:len(a)])


