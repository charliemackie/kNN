# kNN Basics
Leveraging kNN algorithm's to classify different data sets 

# Iris

There is a wieghted and unweighted method of determining which class a certain Iris should belong to.

In the unweighted method, the closest neighbors determine the classification of set instance. (Most common)

In the weighted method, the distance of each neighbor is assessed and given a weight which will help determine the classification of set instance.

# SkLearn

This is a simple interface for using SkLearn for a kNN calculation

weights = 'uniform' assigns uniform weights to each neighbor. This is also the default value.

weights = 'distance' assigns weights proportional to the inverse of the distance from the query sample.

## Parameters:

  1. n_neighbors, =k - the number of neighbors used for the method
  2. weights, 'uniform' or 'distance' 
  3. algorithm, change the efficiency 
  4. leaf_size, default = 30
  5. p, distance metric
  6. metric, 'minkowski' the distance metric to use for the tree
  7. n_jobs, CPU core efficiency ... multithreading?


Adapted from: https://www.python-course.eu/k_nearest_neighbor_classifier.php
