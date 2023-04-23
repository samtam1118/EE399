## Analysis and Classification of Handwritten Digits Using SVD, Linear Discriminant Analysis, SVM, and Decision Trees
## EE399 HW3 SP23 Sam Tam
## Abstract
This project presents an analysis and classification of the MNIST dataset, which consists of handwritten digits. The analysis involves performing a Singular Value Decomposition (SVD) of the digit images and exploring the singular value spectrum to determine the necessary number of modes for good image reconstruction. The interpretation of the U, Σ, and V matrices in the SVD is also discussed.

For classification, Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Trees are used to identify individual digits in the training set. Two pairs of digits are selected for classification: one pair that is difficult to separate and one pair that is easy to separate. The performance of each classifier is evaluated on both the training and test sets.

Finally, the SVM and decision tree classifiers are used to classify all ten digits in the MNIST dataset. The accuracy of the classifiers is compared, and it is shown that SVM is more effective for this task than decision trees. Overall, this project provides an overview of various techniques for analyzing and classifying handwritten digits, which has important applications in areas such as optical character recognition and digitized document processing.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Implementation](#algorithm-implementation)
- [Computational Results](#computational-results)
- [Summary and Conclusions](#summary-and-conclusions)

## Introduction

Handwritten digit recognition is a fundamental problem in the field of computer vision and pattern recognition. The MNIST dataset, consisting of images of handwritten digits, has been widely used as a benchmark for testing various image recognition algorithms. In this project, we analyze and classify the MNIST dataset using various techniques, including Singular Value Decomposition (SVD), Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Decision Trees.

First, we perform an SVD analysis of the digit images to understand the underlying structure of the dataset. We examine the singular value spectrum and determine the necessary number of modes for good image reconstruction. We also discuss the interpretation of the U, Σ, and V matrices in the SVD.

Next, we use LDA, SVM, and Decision Trees to classify individual digits in the dataset. We select two pairs of digits to classify: one pair that is difficult to separate and one pair that is easy to separate. We evaluate the performance of each classifier on both the training and test sets to compare their effectiveness.

Finally, we compare the accuracy of SVM and decision tree classifiers in separating all ten digits in the MNIST dataset. We discuss the limitations and advantages of each method, and highlight the importance of analyzing and classifying handwritten digits for applications such as optical character recognition and digitized document processing.

Overall, this project provides an overview of various techniques for analyzing and classifying handwritten digits, which is a challenging and important problem in the field of computer vision and pattern recognition.

## Theoretical Background

### Singular Value Decomposition (SVD):

Singular Value Decomposition (SVD) is a factorization method used to decompose a matrix into three components: U, Σ, and V. Here, U is an m x m orthogonal matrix, Σ is an m x n diagonal matrix, and V is an n x n orthogonal matrix. In image processing, SVD is used for image compression and noise reduction.

### Linear Discriminant Analysis (LDA):

Linear Discriminant Analysis (LDA) is a technique used for dimensionality reduction and classification. LDA finds a linear combination of features that maximally separates two or more classes in the data. LDA is commonly used in machine learning for feature extraction and classification.

### Support Vector Machines (SVM):

Support Vector Machines (SVM) is a supervised learning method used for classification and regression analysis. SVM finds a hyperplane that maximally separates two or more classes in the data. SVM is a popular technique in machine learning for its ability to handle high-dimensional data and non-linear boundaries.

### Decision Trees:

Decision Trees are a popular technique in machine learning used for classification and regression analysis. Decision Trees create a model that predicts the value of a target variable based on several input variables. The model is represented as a tree, where each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label.

In this project, we use these techniques to analyze and classify handwritten digits in the MNIST dataset. SVD is used to decompose the digit images into their underlying structure, while LDA, SVM, and Decision Trees are used to classify the digits based on their features.

## Algorithm Implementation

## Computational Results

## Summary and Conclusions

