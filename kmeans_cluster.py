#original file can be found in PySpark Documentation

#k-means clustering: https://en.wikipedia.org/wiki/K-means_clustering

from numpy import array
from math import sqrt

from pyspark.mllib.clustering import KMeans, KMeansModel

# Load the data as a SparkContext object
data = sc.textFile("data/mllib/kmeans_data.txt")

# Parse the data as a numpy array using list comprehension
parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# Cluster the data by training the model to find the best
# 2 centroids over 10 iterations of the parsed data

clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode="random")

# Evaluate clustWithin Set Sum of Squared Errors

# Compute the Within Set Sum of Squared Errors 
# Steps to Find WSSE:
# 1. Square the distance of one point from its respective centroid 
# 2. Summate all the rest of the points distances from that centroid
# 3. Do the same for the next cluster and its centroid and so on
# 4. Summate the totals of the each respective clusters' squared errors
# 5. The WSSE has been found; note that a lower number tends to mean
#    the centroids are more tigtly linked with their respective points
#    that does not necessarily mean that the model is better though 
#    depending on context.

#function to calculate squared error of one point from its respective centroid
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

#add together all the squared errors and print the result
WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# Save and load model
clusters.save(sc, "hdfs///user/lev/KMeansModel_#1")
loadModel = KMeansModel.load(sc, "hdfs///user/lev/KMeansModel_#1")
