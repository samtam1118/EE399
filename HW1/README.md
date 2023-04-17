# Least-Squares Fitting Error
## EE399 SP23 Sam Tam
## Abstract
This assignment focuses on fitting a model to given data using the method of least-squares error. The model to be fitted is of form f(x) = A cos(Bx) + Cx + D, where A, B, C, and D are parameters to be determined. 
The objectives of the assignment are four-fold: (i) finding the minimum error and determining the optimal parameter values; (ii) generating a 2D error landscape by fixing two parameters and sweeping through the other two; (iii) fitting different polynomial models (line, parabola, 19th degree) to training data and evaluating their errors on training and test data; and (iv) repeating the process with different training and test data. Python programming is used for implementing the algorithms and visualizing the results.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Implementation](#algorithm-implementation)
- [Computational Results](#computational-results)
- [Summary and Conclusions](#summary-and-conclusions)

## Introduction

This assignment focuses on fitting a model to a given dataset using the least-squares error method, and then analyzing the model by fixing and sweeping through its parameters. The dataset comprises 31 data points, and the model is a combination of a cosine function, a linear function, and a constant. The code written for this assignment aims to find the minimum error and determine the corresponding values of the parameters A, B, and C that correspond to this error. Once the optimal parameter values are obtained, two of the parameters are fixed while the remaining two are swept through various values to generate a 2D loss landscape. All possible combinations of fixed and swept parameters are examined, and the number of minima found during the parameter sweep is reported.

In addition to analyzing the model in this way, the first 20 data points are also used as training data to fit a line, a parabola, and a 19th degree polynomial to the data. The least-squares error for each model is computed over the training points, and the performance of these models is also evaluated using the remaining 10 test data points. This process is repeated using the first 10 and last 10 data points as training data, and the model is fit to the remaining middle 10 data points. Finally, the results of these experiments are compared to provide a comprehensive exploration of how to fit a model to data and analyze the resulting model using least-squares error and parameter sweeping. Through this assignment, a thorough understanding of model fitting and analysis techniques will be gained, providing valuable insights for practical applications in data analysis and modeling.

## Theoretical Background

This assignment utilizes several key concepts in data modeling and analysis, including least-squares error, parameter sweeping, and model evaluation.

- Least-squares error: Least-squares error is a commonly used method for fitting a model to data. It involves finding the parameters of the model that minimize the sum of squared residuals between the model's predictions and the actual data points. In other words, it seeks to minimize the discrepancy between the model's predictions and the observed data. This assignment aims to find the optimal values of the parameters A, B, and C that minimize the least-squares error for the given dataset, which consists of 31 data points.
![image](https://user-images.githubusercontent.com/110360994/231066298-c593fc27-f65d-49df-bec7-d0405b05b6c5.png) `equation of least-squares error`

- Parameter sweeping: Parameter sweeping is a technique used to systematically explore the effects of different parameter values on the performance of a model. This assignment, after obtaining the optimal parameter values using the least-squares error method, two of the parameters are fixed while the remaining two are swept through various values. This generates a 2D loss landscape, where the loss function (e.g., the least-squares error) is plotted against the two swept parameters. By examining the number of minima found during the parameter sweep, insights can be gained into the sensitivity of the model's performance to different parameter values.

- Model evaluation: Model evaluation is a critical step in the data modeling process to assess the performance of the fitted model. In this assignment, it evaluates the performance of the fitted model by computing the least-squares error over the training data points for different models, including a line, a parabola, and a 19th degree polynomial. It also evaluates the performance of these models on the test data points, which are held out from the training data, to assess their generalization performance. By comparing the performance of different models, insights can be gained into their accuracy and suitability for the given dataset.

At the end, this assignment employs the theoretical concepts of least-squares error, parameter sweeping, and model evaluation to fit a model to the given dataset and analyze its performance. These concepts are fundamental in data modeling and analysis, providing a solid foundation for understanding and interpreting the results obtained from the code.

## Algorithm Implementation

### Least-Squares Error Fitting
The first step in the code implementation is to fit a model to the given dataset using the least-squares error method. The dataset consists of 31 data points, and the model is a combination of a cosine function, a linear function, and a constant.

- Load the dataset into the code and extract the input features and target variable.
- Define the model function as a combination of the cosine function, linear function, and constant term.
- Define a loss function that calculates the sum of squared residuals between the model's predictions and the actual target values.
- Use an optimization algorithm (e.g., gradient descent) to find the optimal values of the parameters A, B, and C that minimize the loss function.
- Store the optimal parameter values for further analysis.
- Here's code of the finding error
```
def velfit(c, x, y):
    e2 = np.sqrt(np.sum((c[0]*np.cos(c[1]*x)+c[2]*x+c[3]-y)**2)/len(x))
    return e2

v0 = np.array([3, 1*np.pi/4, 2/3, 32])

res= opt.minimize(velfit, v0, args=(X, Y), method='Nelder-Mead')

Minimum error: {:.2f}".format(res.fun)
```

### Parameter Sweeping
After obtaining the optimal parameter values using the least-squares error method, the code performs parameter sweeping to explore the effects of different parameter values on the model's performance.

- Choose two parameters to be fixed, and two parameters to be swept through various values.
- Generate a 2D loss landscape by sweeping the two parameters through a range of values and computing the loss function for each combination of parameter values.
- Plot the loss landscape to visualize the performance of the model with different parameter values.
- Count the number of minima found during the parameter sweep to assess the sensitivity of the model's performance to different parameter values.
- Repeat the parameter sweeping process for different combinations of fixed and swept parameters to thoroughly analyze the model's behavior.
- Here's one example code of the sweeping
```
B_vals = np.linspace(0, 0.5, 50)
C_vals = np.linspace(-5, 5, 50)
A_vals = np.zeros((len(B_vals), len(C_vals)))
D_vals = np.zeros((len(B_vals), len(C_vals)))
for i, B in enumerate(B_vals):
    for j, C in enumerate(C_vals):
        params = [c[0], B, C, c[3]]
        A_vals[i, j] = velfit(params, X, Y)
        D_vals[i, j] = c[3]
```

### Model Evaluation
Once the optimal model parameters and insights from parameter sweeping are obtained, the code evaluates the performance of the model on the dataset.

- Fit different models (e.g., a line, a parabola, and a 19th degree polynomial) to the training data points.
- Compute the least-squares error for each model over the training data points to assess their performance.
- Evaluate the performance of the models on the test data points, which are held out from the training data, to assess their generalization performance.
- Compare the performance of different models to determine their accuracy and suitability for the given dataset.
```
poly_coeffs = np.polyfit(X_train, Y_train, 19)
poly_fit = np.polyval(poly_coeffs, X_train)
poly_error_train = np.sqrt(np.mean((Y_train - poly_fit)**2))
poly_fit_test = np.polyval(poly_coeffs, X_test)
poly_error_test = np.sqrt(np.mean((Y_test - poly_fit_test)**2))
```
The algorithm implementation and development section provides an overview of the steps involved in fitting the model to the dataset, performing parameter sweeping, and evaluating the model's performance. It gives a clear understanding of the approach taken in the code and the key concepts utilized in the implementation.

## Computational Results

### Part I

The code uses the numpy and scipy.optimize libraries to fit a velocity model to given data points. The data points are stored in the X and Y arrays, representing the independent and dependent variables, respectively. The velfit() function is defined to calculate the error between the model and the data using a set of parameters (c) and the data arrays (x and y). The opt.minimize() function from scipy.optimize is then used to minimize the error by adjusting the parameters. The initial guess for the parameters is set to v0 = np.array([3, 1*np.pi/4, 2/3, 32]).

After fitting the model, the resulting parameters (c) are used to generate a smooth curve (y_fit) using a higher resolution array x2 for plotting. The original data points and the fitted curve are then plotted using matplotlib.pyplot with black circles and a red line, respectively. The minimum error between the model and the data is calculated and printed.

#### Result
The minimum error between the model and the data is found to be 1.59. The values of the fitted parameters are:

A = 2.17
B = 0.91
C = 0.73
D = 31.45

### Part II

The code performs a sweep through different values of the parameters A, B, C, and D to analyze the loss landscape of the model. Multiple 2D plots are generated using matplotlib.pyplot.pcolor() to visualize the error values (loss) for different combinations of the parameters. The color of each point in the plot represents the error value at that point, with a colorbar for reference.

Six different plots are generated, each with two parameters fixed at the fitted values from Part I and sweeping through different values of the other two parameters. The plots are:

Loss Landscape for A and B fixed, with C and D as variables.
Loss Landscape for A and C fixed, with B and D as variables.
Loss Landscape for A and D fixed, with B and C as variables.
Loss Landscape for B and C fixed, with A and D as variables.
Loss Landscape for B and D fixed, with A and C as variables.
Loss Landscape for C and D fixed, with A and B as variables.
#### Result
The loss landscapes reveal the sensitivity of the model to different parameter values. From the plots, it can be observed that the error values (loss) vary across different parameter combinations, indicating the presence of local minima or maxima in the loss landscape. This suggests that the model's performance may be sensitive to the initial parameter values and may require careful tuning during optimization. Further analysis and experimentation may be needed to determine the optimal parameter values for the model.

![image](https://user-images.githubusercontent.com/110360994/231019110-580e4386-714e-4a66-a090-6dba2c8da147.png) `Sweep through values of A and B`
![image](https://user-images.githubusercontent.com/110360994/231019133-ce1f4152-fcc4-4610-bc4b-c5e3c3c7263f.png) `Sweep through values of A and C`
![image](https://user-images.githubusercontent.com/110360994/231019301-54f9ddd1-71d3-43d7-9eb4-2b7b03421bc2.png) `Sweep through values of A and D`
![image](https://user-images.githubusercontent.com/110360994/231020449-5bc079fb-5011-401b-9b27-c8b969b234de.png) `Sweep through values of B and C`
![image](https://user-images.githubusercontent.com/110360994/231020523-938f6344-293c-4fd2-b4df-cdefb806b732.png) `Sweep through values of B and D`
![image](https://user-images.githubusercontent.com/110360994/231020556-fccbc626-d0e3-477f-8216-21ab42ea3e5d.png) `Sweep through values of C and D`

### Part III
In this part of the code, the data is split into training and test sets. Three different models, namely a line, a parabola, and a 19th degree polynomial, are fitted to the training data. The least square errors are calculated for each model on both the training and test data. The data and the fitted models are also plotted.

#### Result

Line Error: 2.24
Parabola Error: 2.13
19th Degree Polynomial Error: 0.03

Test Errors:
Line Error: 3.36
Parabola Error: 8.71
19th Degree Polynomial Error: 28626352734.19
![image](https://user-images.githubusercontent.com/110360994/231062947-f1dbbddb-2530-4104-b75d-d259a67a656f.png)

From the results, it can be observed that the 19th degree polynomial has the lowest training error, indicating that it fits the training data very well. However, when tested on the held-out test data, the 19th degree polynomial has the highest error, suggesting that it may be overfitting the training data and not generalizing well to new data. On the other hand, the line and parabola models have similar training errors, but the parabola has a slightly lower test error, indicating that it may generalize better to new data compared to the line model.

### Part IV
In this part of the code, the data is split into training and testing data in a different way compared to Part iii. The first 10 and last 10 data points are used as training data, while the middle 10 data points are used as test data. The same three models (line, parabola, and 19th degree polynomial) are fitted to the training data, and the errors are calculated for both training and test data. The results are plotted as well.

#### Result

Training errors:

Line fit: 68.57
Parabola fit: 68.51
19th degree poly fit: 0.54

Test errors:
Line fit: 86.45
Parabola fit: 84.44
19th degree poly fit: 2575325.02
![image](https://user-images.githubusercontent.com/110360994/231062917-e8f50d19-62a1-4166-befc-17f60e2709c9.png)

Similar to Part iii, the 19th degree polynomial has the lowest training error, indicating that it fits the training data very well. However, when tested on the held-out test data, it has the highest error, suggesting overfitting. The line and parabola models have higher errors compared to Part iii, both in training and testing, which indicates that using the first 10 and last 10 data points as training data may not be optimal for these models. However, the relative performance of the models is similar to that in Part iii, with the parabola model having slightly lower errors compared to the line model.


## Summary and Conclusions

In this code, we implemented a velocity fitting function velfit to fit a model to a set of data points. We used the Nelder-Mead optimization method from the scipy.optimize module to minimize the error between the model and the data. We then plotted the data points along with the fitted model using matplotlib.

We further analyzed the loss landscape of the model by sweeping through different values of the parameters A, B, C, and D while keeping some of them fixed. We generated contour plots using pcolor function from matplotlib to visualize the loss landscape.

- The main conclusions from the analysis are as follows:

- The minimum error between the model and the data was calculated to be res.fun, which provides a measure of how well the model fits the data.

- The optimized parameter values for A, B, C, and D were obtained as c[0], c[1], c[2], and c[3] respectively, after fitting the model to the data.

- By analyzing the contour plots of the loss landscape with fixed values of parameters, we observed that changing the values of C and D has a significant impact on the model's performance, as seen in the contour plots for A and B fixed, A and C fixed, and A and D fixed. This suggests that C and D are important parameters to consider in the model fitting process.

- We also observed that changing the values of B and C has a relatively smaller impact on the model's performance, as seen in the contour plot for B and C fixed. This suggests that B and C may not have as strong of an influence on the model's performance compared to A and D.

- Overall, the contour plots provide insights into the sensitivity of the model to different parameter values and can help in understanding the trade-offs and interactions between the parameters.

In conclusion, this code provides a framework for fitting a model to data, optimizing the parameters, and analyzing the loss landscape to gain insights into the model's performance. Further experimentation with different parameter values can help in fine-tuning the model and improving its accuracy in fitting the data.
