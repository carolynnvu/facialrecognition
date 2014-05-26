import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import random
from array import *

def visualize(scores, faces):
  """
  The function for visualization part, 
  which put the image at the coordinates given by their coefficients of 
  the first two principal components (with translation and scaling).

  scores: n x 2 array, where each row contains the first 2 principal component scores of each face
  faces: n x 4096 array
  """
  pc_min, pc_max = np.min(scores, 0), np.max(scores, 0)
  pc_scaled = (scores - pc_min) / (pc_max - pc_min)  
  fig, ax = plt.subplots()
  for i in range(len(faces)):
    imagebox = offsetbox.OffsetImage(faces[i, :].reshape(64,64).T, cmap=plt.cm.gray, zoom=0.5)
    box = offsetbox.AnnotationBbox(imagebox, pc_scaled[i, 0:2])
    ax.add_artist(box)
  plt.show()

# Example code starts from here
# Load the data set
faces = sp.genfromtxt('faces.csv', delimiter=',')

# Example for displaying the first face, which may help you how the data set presents
#plt.imshow(faces[0, :].reshape(64, 64).T, cmap=plt.cm.gray)
#plt.show()

# Your code starts from here ....

# a. Randomly display a face in range (0, 399): 0 <= random_face <= 399
# STUDENT CODE TODO

print "Displaying a random face...\n"
random_face = random.randint(0,399)
plt.imshow(faces[random_face, :].reshape(64, 64).T, cmap=plt.cm.gray)
plt.show()

# b. Compute and display the mean face
# STUDENT CODE TODO

print("Computing and displaying the mean of the faces...\n")
mean = [0]*4096 #Initialize mean array

#Sum all the 400 images' 4096 pixels
for j in range(400):
 mean = np.add(mean, faces[j])

#Divide by the total number of images: 400
m = 400
for i in range(len(mean)):
  mean[i] = np.divide(mean[i], m)

mean_face = mean
plt.imshow(mean_face.reshape(64, 64).T, cmap=plt.cm.gray)
plt.show()

# c. Centralize the faces by substracting the mean
# STUDENT CODE TODO

"""Subtract mean from face images"""
for k in range(400):
  faces[k] = np.subtract(faces[k], mean)

"""Construct centralized data matrix X of dimensions 400 by 4096"""
X = np.zeros((400, 4096))

for i in range(400):
  X[i] = faces[i]

X = np.matrix(X)

# d. Perform SVD (you may find scipy.linalg.svd useful)
# STUDENT CODE TODO

U, s, V_t = np.linalg.svd(X)

# e. Show the first 10 priciple components
# STUDENT CODE TODO

"""The principal components are the columns of V.
#But this is the same as the rows of V_t, so for
#simplicity we will work with the rows of V_t.
#We will now get the first ten rows of V_t and reshape
#each row of length 4096 to a 64 by 64 matrix."""

component1 = V_t[0].reshape(64, 64)
component2 = V_t[1].reshape(64, 64)
component3 = V_t[2].reshape(64, 64)
component4 = V_t[3].reshape(64, 64)
component5 = V_t[4].reshape(64, 64)
component6 = V_t[5].reshape(64, 64)
component7 = V_t[6].reshape(64, 64)
component8 = V_t[7].reshape(64, 64)
component9 = V_t[8].reshape(64, 64)
component10 = V_t[9].reshape(64, 64)

print "Displaying images of the first 10 principcal components..."

"""All 10 images will appear, but not altogether at once. 
#The only way to see each image is to ex out of the current one
#on display and then the next one will appear."""
print "Image 1"
plt.imshow(component1.T, cmap=plt.cm.gray)
plt.show()
print "Image 2"
plt.imshow(component2.T, cmap=plt.cm.gray)
plt.show()
print "Image 3"
plt.imshow(component3.T, cmap=plt.cm.gray)
plt.show()
print "Image 4"
plt.imshow(component4.T, cmap=plt.cm.gray)
plt.show()
print "Image 5"
plt.imshow(component5.T, cmap=plt.cm.gray)
plt.show()
print "Image 6"
plt.imshow(component6.T, cmap=plt.cm.gray)
plt.show()
print "Image 7"
plt.imshow(component7.T, cmap=plt.cm.gray)
plt.show()
print "Image 8"
plt.imshow(component8.T, cmap=plt.cm.gray)
plt.show()
print "Image 9"
plt.imshow(component9.T, cmap=plt.cm.gray)
plt.show()
print "Image 10"
plt.imshow(component10.T, cmap=plt.cm.gray)
plt.show()

print "\n"


# f. Visualize the data by using first 2 principal components using the function "visualize"
# STUDENT CODE TODO

"""Compute matrix W, where W = XV.
We compute matrix multiplication of X and V"""

W = X*(np.transpose(V_t))

"""Create matrix SCORE"""
SCORE = np.zeros((30, 2))

"""Selecting 30 random faces"""
random_thirty = [0]*30
matrixThirtyFaces = np.zeros((30, 4096))

for i in range(30):
  random_thirty[i] = random.randint(0, 399)
  SCORE[i] = W[random_thirty[i], 0:2]
  matrixThirtyFaces[i] = faces[random_thirty[i], :]

"""Pass parameters into visualize function"""
print "Now visualizing using first 2 principal components on 30 random faces...\n"
visualize(SCORE, matrixThirtyFaces)

# g. Plot the proportion of variance explained
# STUDENT CODE TODO

"""Compute the total variance"""
TotalVar = 0.0
Diagonal_of_S = np.array(s)

for i in range(len(Diagonal_of_S)):
  TotalVar = np.add(TotalVar, Diagonal_of_S[i])

"""Computer the variance explained for the first 10 principal components"""
PC_Var = [0.0]*10

for j in range(10):
  PC_Var[j] = Diagonal_of_S[j]/TotalVar

print "Plotting the proportion of variance explained...\n"
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Proportion of Variance')
plt.plot(PC_Var)
plt.show()

# h. Face reconstruction using 5, 10, 25, 50, 100, 200, 300, 399 principal components
# STUDENT CODE TODO

print "Reconstructing the randomly chosen face...\n"
"""Choose a random face"""
i = random.randint(0, 399)
k_sum_vector = [0]*4096
sub = np.array(W[i])


print "Reconfiguring with k=5 principal components"
k = 5
for j in range(k):
  k_sum_vector = np.add(k_sum_vector, np.multiply(sub[0][j], V_t[j]))

x_hat = np.add(mean, k_sum_vector)
plt.imshow(x_hat.reshape(64, 64).T, cmap=plt.cm.gray)
plt.show()

print "Reconfiguring with k=10 principal components"
k_sum_vector = [0]*4096
k = 10
for j in range(k):
 k_sum_vector = np.add(k_sum_vector, np.multiply(sub[0][j], V_t[j]))
 
x_hat = np.add(mean, k_sum_vector)
plt.imshow(x_hat.reshape(64, 64).T, cmap=plt.cm.gray)
plt.show()

print "Reconfiguring with k=25 principal components"
k_sum_vector = [0]*4096
k = 25
for j in range(k):
 k_sum_vector = np.add(k_sum_vector, np.multiply(sub[0][j], V_t[j]))
 
x_hat = np.add(mean, k_sum_vector)
plt.imshow(x_hat.reshape(64, 64).T, cmap=plt.cm.gray)
plt.show()

print "Reconfiguring with k=50 principal components"
k_sum_vector = [0]*4096
k = 50
for j in range(k):
 k_sum_vector = np.add(k_sum_vector, np.multiply(sub[0][j], V_t[j]))
 
x_hat = np.add(mean, k_sum_vector)
plt.imshow(x_hat.reshape(64, 64).T, cmap=plt.cm.gray)
plt.show()

print "Reconfiguring with k=100 principal components"
k_sum_vector = [0]*4096
k = 100
for j in range(k):
 k_sum_vector = np.add(k_sum_vector, np.multiply(sub[0][j], V_t[j]))
 
x_hat = np.add(mean, k_sum_vector)
plt.imshow(x_hat.reshape(64, 64).T, cmap=plt.cm.gray)
plt.show()

print "Reconfiguring with k=200 principal components"
k_sum_vector = [0]*4096
k = 200
for j in range(k):
 k_sum_vector = np.add(k_sum_vector, np.multiply(sub[0][j], V_t[j]))
 
x_hat = np.add(mean, k_sum_vector)
plt.imshow(x_hat.reshape(64, 64).T, cmap=plt.cm.gray)
plt.show()

print "Reconfiguring with k=300 principal components"
k_sum_vector = [0]*4096
k = 300
for j in range(k):
 k_sum_vector = np.add(k_sum_vector, np.multiply(sub[0][j], V_t[j]))
 
x_hat = np.add(mean, k_sum_vector)
plt.imshow(x_hat.reshape(64, 64).T, cmap=plt.cm.gray)
plt.show()

print "Reconfiguring with k=399 principal components"
k_sum_vector = [0]*4096
k = 399
for j in range(k):
 k_sum_vector = np.add(k_sum_vector, np.multiply(sub[0][j], V_t[j]))
 
x_hat = np.add(mean, k_sum_vector)
plt.imshow(x_hat.reshape(64, 64).T, cmap=plt.cm.gray)
plt.show()


# i. Plot the reconstruction error for k = 5, 10, 25, 50, 100, 200, 300, 399 principal components
#    and the sum of the squares of the last n-k (singular values)
#    [extra credit]
# STUDENT CODE TODO
