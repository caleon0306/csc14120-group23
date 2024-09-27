from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn import metrics

from yellowbrick.cluster import KElbowVisualizer

import numpy as np
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

#gets predictions with correct labels
def getPredictions(model, true, X, numK):
    model.fit(X)
    return mapLabels(model.predict(X), true, numK)

#since the labels get swapped sometimes a mapping function will swap incorrect labels
#totheir correct position
def mapLabels(predictions, true, numK):
    #variable to track if all groups are found
    labels = 0
    #list to track what labels are found
    foundLabels = []
    #dic to map mislabled data to their correct label
    maping = {}
    for i in range(len(predictions)):
        #if an unseen label is found map it to its correct label
        if predictions[i] not in foundLabels:
            maping[predictions[i]] = true[i]
            foundLabels.append(predictions[i])
            labels += 1
        if labels == numK:
            break
    #map incorrect labels to correct label
    for i in range(len(predictions)):
        predictions[i] = maping[predictions[i]]
    return predictions
        

"""
def findDistortion(X,centers):
    return (sum(np.min(cdist(X, centers),axis=1))/X.shape[0])

def plotGraph(distortions):
    plt.plot(range(MIN_K, MAX_K+1), distortions,'bx-')
    plt.xlim(MIN_K-1, MAX_K+1)
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Finding Best k')

distortions = []
for k in range(MIN_K, MAX_K + 1):
    distortions.append(findDistortion(X,findCenters(X,k)))

plotGraph(distortions)

"""

#constants for min and max k range
MIN_K = 1
MAX_K = 10
NUM_CLUSTERS = 4

#create sample data
X, y_true = make_blobs(n_samples=300, centers=NUM_CLUSTERS,
                       cluster_std=0.60, random_state=0)


#TODO find best K
#n_init set to suppress warning
model = KMeans(n_init = 10)
visualizer = KElbowVisualizer(model, k=(MIN_K,MAX_K+1))

visualizer.fit(X)
visualizer.show()

bestK = visualizer.elbow_value_
print("The Best K:", bestK)


# TODO calculate accuracy for best K
model = KMeans(n_clusters = bestK,n_init = 10)
model.fit(X)
predictions = getPredictions(model, y_true, X, bestK)

correct = 0
for x in range(len(predictions)):
    if predictions[x] == y_true[x]:
        correct += 1
print("Accuracy: " + str(correct/len(predictions) * 100) + "%")


# TODO draw a confusion matrix
confusionMatrix = metrics.confusion_matrix(y_true, predictions)
cmDisplay = metrics.ConfusionMatrixDisplay(confusion_matrix = confusionMatrix)
cmDisplay.plot()
plt.title("Confusion Matrix")