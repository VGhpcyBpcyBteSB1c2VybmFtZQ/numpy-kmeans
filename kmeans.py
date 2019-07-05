from PIL import Image
import numpy as np
import random


img = np.array(Image.open("img.jpg"))
test = np.zeros(img.shape)
runs = 100

k = 4
centroids = np.zeros((k, 3))
distances = np.zeros((k, img.shape[0], img.shape[1]))
labels = np.zeros((img.shape[0], img.shape[1]))


# creating the initial centroids
for i in range(0, k):
    centroids[i] = img[int(random.random() * img.shape[0]), int(random.random() * img.shape[1])]


# K means algorithm
for r in range(0, runs):

    # calculate the distances
    for i in range(0, k):
        random.seed()
        distances[i] = np.linalg.norm(img - centroids[i], axis=2)

    # find the labels for each point
    newLabels = np.argmin(distances, axis=0)

    if ((newLabels == labels).all()):  # break the loop if the clusters have converged
        break
    else:
        labels = newLabels
        # update the centroids
        for i in range(0, k):
            centroids[i] = np.mean(img[labels == i], axis=0)

img[newLabels == 0] = [255, 0, 0]
img[newLabels == 1] = [0, 255, 0]
img[newLabels == 2] = [0, 0, 255]
img[newLabels == 3] = [0, 255, 255]

Image.fromarray(img).save("out.png")
print("Done")
