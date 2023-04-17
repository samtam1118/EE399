## Image Correlation, Eigenvectors, and Principal Component Analysis
## EE399 HW2 SP23 Sam Tam
## Abstract
This homework assignment presents an analysis of image correlation, eigenvectors, and principal component analysis (PCA). The first section introduces the topic and provides an overview of the research. The theoretical background section discusses the concepts of correlation, eigenvectors, and PCA. The algorithm implementation and development section outlines the steps to compute the correlation matrix, find eigenvectors and principal component directions, and compare them. The computational results section presents the findings, including plots of correlation matrices, images with high and low correlation, and the percentage of variance captured by SVD modes. Finally, the summary and conclusions section highlights the key findings and discusses their implications in image analysis and processing.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Implementation](#algorithm-implementation)
- [Computational Results](#computational-results)
- [Summary and Conclusions](#summary-and-conclusions)

## Introduction

Image analysis and processing are fundamental tasks in various fields such as computer vision, pattern recognition, and image recognition. One important aspect of image analysis is understanding the relationships between images, which can be achieved through the computation of image correlation. Image correlation measures the similarity or dissimilarity between two images, and it has wide-ranging applications, including image matching, image registration, and image retrieval.

In this homework assignment, we explore the concept of image correlation and its applications. We start by computing a correlation matrix for a set of images using dot product (correlation) between image columns in a matrix. We then analyze the correlation matrix to identify the most highly correlated and most uncorrelated image pairs. Additionally, we investigate the computation of eigenvectors and principal component directions through the Singular Value Decomposition (SVD) of a matrix. We compare the eigenvectors and principal component directions, and compute the percentage of variance captured by each of the first 6 SVD modes.

This homework assignment aims to provide a comprehensive understanding of image correlation, eigenvectors, and principal component analysis (PCA) in the context of image analysis and processing. Theoretical concepts will be discussed, and practical implementations will be demonstrated using Python programming. The computational results will be presented and discussed, highlighting the insights gained from the analysis.

## Theoretical Background

Image correlation is a measure of similarity or dissimilarity between two images, which quantifies the degree to which one image resembles another. It is commonly used in various image processing tasks such as image matching, image registration, and image retrieval.

The correlation between two images can be computed using the dot product (correlation) between the image columns in a matrix. For a given set of images represented as a matrix X, the correlation matrix C can be computed as C = X^T * X, where X^T is the transpose of the image matrix X. The element c_jk in the correlation matrix C represents the correlation between the jth and kth images, and it is computed as the dot product between the jth and kth image columns in X.

Eigenvectors and eigenvalues are important concepts in linear algebra, which play a significant role in image processing and analysis. Eigenvectors are the directions along which a linear transformation (such as a matrix multiplication) only stretches or compresses the vectors without changing their direction. Eigenvalues represent the scaling factor of the eigenvectors.

The Singular Value Decomposition (SVD) is a popular technique used to analyze the properties of a matrix, including eigenvectors and eigenvalues. SVD decomposes a matrix into three matrices: U, Σ, and V^T, where U and V are orthogonal matrices representing the left and right singular vectors, respectively, and Σ is a diagonal matrix containing the singular values. The singular values represent the square roots of the eigenvalues of the matrix X^T * X or X * X^T, and the singular vectors represent the eigenvectors of X^T * X or X * X^T.

In this homework assignment, we leverage the theoretical background of image correlation, eigenvectors, and SVD to compute correlation matrices, eigenvectors, and principal component directions for a set of images. These concepts provide a solid foundation for the algorithm implementation and development, as well as the analysis of computational results in subsequent sections.

## Algorithm Implementation
The algorithm for computing the correlation matrix, eigenvectors, and principal component directions for a set of images was implemented and developed as follows:

### Part A
Computing the Correlation Matrix: The dot product (correlation) between the first 100 images in the matrix X was computed to obtain a 100x100 correlation matrix C. Each element of the correlation matrix C was computed as cjk = xTj xk, where xj is the jth column of the matrix X. The correlation matrix C was then plotted using the pcolor function, which provided a visual representation of the similarity or dissimilarity between pairs of images.
```
X_100 = X[:, :100]  # Select the first 100 columns of X
C = np.dot(X_100.T, X_100)  # Compute the dot product of X_100 with its transpose
```
### Part B 
Highly Correlated and Uncorrelated Images: Analysis of the correlation matrix C was performed to identify the pairs of images that were most highly correlated and most uncorrelated. The pairs were identified based on the maximum and minimum values in C, respectively, and their corresponding image indices. The images corresponding to these pairs were then plotted to visualize the highly correlated and uncorrelated image pairs.
```
# Find the indices of the most highly correlated and most uncorrelated images
i, j = np.unravel_index(np.argmax(C - np.eye(C.shape[0])*np.max(C)), C.shape)
k, l = np.unravel_index(np.argmin(C + np.eye(C.shape[0])*np.max(C)), C.shape)
```
### Part C
Computing the 10x10 Correlation Matrix: Similar to part (a), a 10x10 correlation matrix was computed between the images using the same dot product (correlation) approach. Each element of the 10x10 correlation matrix was computed as cjk = xTj xk, where xj is the jth column of the matrix X. The correlation matrix was then plotted to visualize the similarity or dissimilarity between pairs of images.
```
# Extract the corresponding image columns from X
X_10 = X[:, image_indices]
# Compute the correlation matrix
C_10 = np.dot(X_10.T, X_10)  # Compute the dot product of X_10 with its transpose
```
### Part D
Computing Matrix Y and Eigenvectors: The matrix Y = X * X^T was computed, where X is the matrix of images. The eigenvalue decomposition of Y was performed to obtain the eigenvectors U and eigenvalues Λ. The eigenvectors were sorted based on the magnitude of their corresponding eigenvalues in descending order, and the first six eigenvectors with the largest magnitude eigenvalues were extracted.
```
# Create the matrix Y = X * X^T
Y = np.dot(X, X.T)
# Compute the eigenvalues and eigenvectors of Y
eigenvalues, eigenvectors = eig(Y)
# Sort the eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]
# Extract the first six eigenvectors with the largest magnitude eigenvalues
eigenvectors_largest = eigenvectors_sorted[:, :6]
# Normalize the eigenvectors
eigenvectors_largest_normalized = eigenvectors_largest / np.linalg.norm(eigenvectors_largest, axis=0)
```
### Part E
SVD of Matrix X and Principal Component Directions: Singular Value Decomposition (SVD) was performed on the matrix X to obtain the left singular vectors U, singular values Σ, and right singular vectors V^T. The first six singular vectors (or principal component directions) were obtained.
```
# Perform Singular Value Decomposition (SVD) on X
U, S, Vt = np.linalg.svd(X, full_matrices=False)
# Find the first six principal component directions
principal_component_directions = U[:, :6]
```
### Part F
Comparing Eigenvectors and SVD Modes: The first eigenvector v1 from part (d) was compared with the first singular vector u1 from part (e) to compute the norm of the difference of their absolute values. This comparison provided insights into the similarity between the two approaches in capturing the image features.
```
# Extract the first eigenvector v1
v1 = eigenvectors_largest_normalized[:, 0]
# Extract the first SVD mode u1
u1 = principal_component_directions[:, 0]
# Compute the norm of difference of absolute values
norm_diff = np.linalg.norm(np.abs(v1) - np.abs(u1))
```
### Part G
Percentage of Variance Captured: The percentage of variance captured by each of the first six singular values (or eigenvalues) was computed by dividing each singular value (or eigenvalue) squared by the sum of all singular values (or eigenvalues) squared. This provided information on the proportion of total variance explained by each mode. The first six SVD modes, along with their corresponding percentage of variance captured, were plotted to visualize their contributions to the total variance.
```
# Perform Singular Value Decomposition (SVD) on X
U, S, VT = np.linalg.svd(X)
# Extract the singular values
singular_values = S[:6]
# Compute the percentage of variance captured by each SVD mode
variance_percentage = (singular_values ** 2) / (np.sum(S ** 2)) * 100
```
Overall, the implemented algorithm provided a comprehensive approach to compute the correlation matrix, eigenvectors, and principal component directions for a set of images, and analyze their properties and contributions to image analysis and processing. The results obtained from the algorithm can be further used for various applications, such as image recognition, classification, and feature extraction.
## Computational Results
### Part A
The 100x100 correlation matrix C was computed by taking the dot product (correlation) between the first 100 images in the matrix X. The resulting correlation matrix C was plotted using the pcolor function, which generated a visual representation of the similarity or dissimilarity between pairs of images.
#### Result
![image](https://user-images.githubusercontent.com/110360994/232350659-1e3510c1-37db-4084-881a-3a02b23ea0ef.png)
### Part B
Analysis of the correlation matrix C revealed that the most highly correlated image pairs were Image 87 and Image 89. On the other hand, the most uncorrelated image pairs were Image 55 and Image 65. These image pairs were plotted to visually inspect the level of similarity or dissimilarity between them.
#### Result
![image](https://user-images.githubusercontent.com/110360994/232350687-24697685-3fbf-4518-b50b-e8c1d1d84f4f.png)
![image](https://user-images.githubusercontent.com/110360994/232350705-bc1cd069-45ca-4e82-9327-40612d2a4488.png)
### Part C
Similar to part (a), a 10x10 correlation matrix was computed between the images in X. The resulting correlation matrix was plotted to visualize the similarity or dissimilarity between pairs of images at a smaller scale.
#### Result
![image](https://user-images.githubusercontent.com/110360994/232350828-0f7e1569-47f9-4860-b764-b2bbce6bb4a7.png)
### Part D
The matrix Y = X * X^T was computed, and its eigenvalue decomposition was performed to obtain the eigenvectors U and eigenvalues Λ. The first six eigenvectors with the largest magnitude eigenvalues were extracted and plotted to visualize their directions in the image space.
#### Result
```
[[ 0.02384327  0.04535378  0.05653196  0.04441826 -0.03378603  0.02207542]
 [ 0.02576146  0.04567536  0.04709124  0.05057969 -0.01791442  0.03378819]
 [ 0.02728448  0.04474528  0.0362807   0.05522219 -0.00462854  0.04487476]
 ...
 [ 0.02082937 -0.03737158  0.06455006 -0.01006919  0.06172201  0.03025485]
 [ 0.0193902  -0.03557383  0.06196898 -0.00355905  0.05796353  0.02850199]
 [ 0.0166019  -0.02965746  0.05241684  0.00040934  0.05757412  0.00941028]]
```
### Part E
Singular Value Decomposition (SVD) was performed on the matrix X to obtain the left singular vectors U, singular values Σ, and right singular vectors V^T. The first six singular vectors (or principal component directions) were obtained and plotted to visualize their directions in the image space.
#### Result
```
[[-0.02384327 -0.04535378 -0.05653196  0.04441826 -0.03378603  0.02207542]
 [-0.02576146 -0.04567536 -0.04709124  0.05057969 -0.01791442  0.03378819]
 [-0.02728448 -0.04474528 -0.0362807   0.05522219 -0.00462854  0.04487476]
 ...
 [-0.02082937  0.03737158 -0.06455006 -0.01006919  0.06172201  0.03025485]
 [-0.0193902   0.03557383 -0.06196898 -0.00355905  0.05796353  0.02850199]
 [-0.0166019   0.02965746 -0.05241684  0.00040934  0.05757412  0.00941028]]
```
### Part F
The first eigenvector v1 from the eigenvalue decomposition of Y was compared with the first singular vector u1 from the SVD of X to compute the norm of the difference of their absolute values. The norm was found to be small, indicating a high level of similarity between the two approaches in capturing the image features.
#### Result
```
Norm of difference between v1 and u1: 6.762462983478808e-16
```
### Part G
The percentage of variance captured by each of the first six singular values (or eigenvalues) was computed, and the results were plotted to visualize the contributions of each mode to the total variance. The plot showed the proportion of total variance explained by each mode, providing insights into the significance of each mode in representing the image data.
#### Result
```
Percentage of Variance Captured by SVD Mode 1: 72.93%
Percentage of Variance Captured by SVD Mode 2: 15.28%
Percentage of Variance Captured by SVD Mode 3: 2.57%
Percentage of Variance Captured by SVD Mode 4: 1.88%
Percentage of Variance Captured by SVD Mode 5: 0.64%
Percentage of Variance Captured by SVD Mode 6: 0.59%
```
![image](https://user-images.githubusercontent.com/110360994/232351259-7fd924a1-cf51-4185-92a2-616a970c20e9.png)

## Summary and Conclusions
In this homework assignment, various computational tasks were performed on a matrix of images to explore their correlations, variability, and representations. The key findings and conclusions from the computational results are summarized as follows:

- Correlation Matrix: The 100x100 correlation matrix was computed by taking the dot product (correlation) between the first 100 images in the matrix X. The resulting correlation matrix was visualized using the pcolor function, providing insights into the similarity or dissimilarity between pairs of images.

- Highly Correlated and Uncorrelated Images: The most highly correlated and uncorrelated image pairs were identified from the correlation matrix, providing information on the level of similarity or dissimilarity between images.

- 10x10 Correlation Matrix: A smaller scale 10x10 correlation matrix was computed to examine the pairwise correlations between images at a reduced resolution, providing a different perspective on the image relationships.

- Eigenvectors and Principal Component Directions: The first six eigenvectors from the eigenvalue decomposition of the matrix Y and the first six singular vectors (or principal component directions) from the SVD of the matrix X were obtained and plotted to visualize the directions of variability in the image space.

- Comparison of Eigenvectors and SVD Modes: The first eigenvector and the first singular vector were compared, and their absolute value difference was computed. The small norm of the difference indicated a high level of similarity between the eigenvalue decomposition and SVD approaches in capturing the image features.

- Percentage of Variance Captured: The percentage of variance captured by each of the first six singular values (or eigenvalues) was computed and plotted to visualize the contributions of each mode to the total variance. This provided insights into the significance of each mode in representing the image data.

In conclusion, the implemented algorithm and computational results provided valuable insights into the correlation, variability, and representation of images in the image space. These findings can be further analyzed and utilized for specific image processing tasks, and can contribute to the development of advanced image analysis and computer vision techniques. The results of this homework assignment can serve as a foundation for future research and applications in the field of image processing and computer vision.
