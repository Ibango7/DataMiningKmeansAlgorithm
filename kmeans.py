import matplotlib.pyplot as plt
import random
import numpy as np 
import pandas as pd
import math as sqrt


# Eucledian distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# mean/average
def compute_mean(clusters, classification):
   return np.average(clusters[classification], axis=0)

# function to visualize clusters
def visualize_clusters(centroids, clusters):
    colors = 10*['b', 'g', 'c','r', 'k']
    for centroid in centroids:
        # plot centroids
        plt.scatter(centroids[centroid][0], centroids[centroid][1],
        marker="x", s=15**2, linewidths=5, label="centroid")
    # plot datapoints and color them based on their cluster
    for classification in clusters:
        color = colors[classification]
        for featureset in clusters[classification]:
            plt.scatter(featureset[0], featureset[1], color=color, s=30)
    plt.title('2008 demographics')
    plt.xlabel('BirthRate(Per1000 - 2008)')
    plt.ylabel('LifeExpectancy(2008)')
    plt.legend(loc='upper right')      
    plt.show()



# read dataset(data2008.csv)
data = pd.read_csv("data2008.csv")
# select features we want to use
DataXy = data[['BirthRate(Per1000 - 2008)','LifeExpectancy(2008)']]
DataXy = np.array(DataXy)


# Implement K means algorithm function
# How it works:
# 1) pick k points as number of clusters
# 2) find the eucledian distance of each point(x,y) in the data set with identified k points-centroids
# 3) assign each data point to the closest centroid using the distance found in the previous step
# 4) find the new centroid by taking the average(mean) in each cluster group
# 5) repeat 2 to 4 for a fixed number of iterations or until centroids do not change.
def k_Means(DataXy, k, max_iter, tolerance = 0.0001):
    # pick centroids randomly
    centroids = {} # a dictionary that stores points (x,y) of centroid
    for i in range(k):
        centroids[i] = random.choice(DataXy)

    # classifications/creating clusters
    #dist = distances(DataXy, centroids)
    for i in range(max_iter):
        clusters = {} # tempcluster
        for i in range(k):
            clusters[i] = [] # key is centroid value is feature set(data points)

        for featureset in DataXy:
            #find the distance between point and cluster(centroid)
            distances = [euclidean_distance(featureset, centroids[centroid]) for centroid in centroids]
            classification = distances.index(min(distances))
            clusters[classification].append(featureset)
        prev_centroids = dict(centroids)

        # redefine centroid in cluster
        # centroid is the mean of all 
        # data points that belong the cluster
        for classification in clusters:
            centroids[classification] = compute_mean(clusters, classification)
            
        # optimization
        # The algorithm is converged if the 
        # percentage change(cnvg) in the centroid values is lower than our accepted
        #  value of tolerance (0.0001)
        optimized = True
        for c in centroids:
            original_centroid = prev_centroids[c]
            current_centroid = centroids[c]
            cnvg = np.sum((current_centroid-original_centroid)/original_centroid*100.00)
            print(f" Sum on each iteration: {cnvg}")
            if cnvg > tolerance:
                optimized = False

        if optimized:
            break

    # vizualizing clusters
    visualize_clusters(centroids, clusters)

    # Some stats on clustering results 
    # number of countries in each cluster
    # list of countries in each cluster
    # The mean life Expectancy and mean Birth Rate for each cluster
    cnt = data['Countries'] # used to query countries in dataset
    i = 1 # label for culster number in loop
    for classification in clusters:
        count = 0 # count number of clusters
        countries = [] # will store countries per clusters
        meanLifExp = [] # mean life expectancy per cluster
        birthRate = [] # birth rate per cluster
        for featureset in clusters[classification]:
            count +=1
            countries.append(cnt[(data['BirthRate(Per1000 - 2008)']== featureset[0]) 
            & (data['LifeExpectancy(2008)'] == featureset[1])])
            # meanLifExp store life expectancies for each cluster to calculate mean later 
            meanLifExp.append(featureset[1])
            # Birthrate for each cluster
            birthRate.append(featureset[0])

        print(f"\n\n=== Number of countries for cluster {i} is {count} ===")
        print(f"Mean life expectancy is {np.mean(meanLifExp):.3f}")
        print(f"Mean birth rate is {np.mean(birthRate):.3f}")
        print(f"=== List  of countries and birth rate for cluster {i} ===\n\n")
        i+=1
        for country, br in zip(countries, birthRate):
            print(f"{country.to_string(index=False)}  {br:.3f}")
# end of k_Means implementation


# Ask user for number of cluster
# and number of iterations
k = int(input("Enter number of cluster you want to generate: "))
n_iter = int(input("Enter maximum number of iterations: "))

# call k-means function 
k_Means(DataXy,k, n_iter)

