## Spatiotemporal Analysis and Forecasting of Sea-Surface Temperature Using LSTM/Decoder Model
## EE399 HW6 SP23 Sam Tam
## Abstract
This homework assignment focuses on the spatiotemporal analysis and forecasting of sea-surface temperature (SST) using an LSTM/decoder model. The goal is to reconstruct SST states and predict sensor measurements based on historical data. The assignment involves training the model, evaluating its performance under different conditions, and analyzing the impact of key variables such as the time lag, noise levels, and number of sensors. The LSTM/decoder model demonstrates promising results in capturing the underlying patterns and variations in the SST data. The analysis sheds light on the optimal time lag, robustness to noise, and the trade-off between the number of sensors and accuracy. The findings contribute to our understanding of spatiotemporal modeling and prediction in SST data and have implications for applications in climate analysis and forecasting.
## Table of Contents
- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Algorithm Implementation](#algorithm-implementation)
- [Computational Results](#computational-results)
- [Summary and Conclusions](#summary-and-conclusions)
## Introduction
Sensor reconstruction and forecasting play a crucial role in numerous fields, including environmental monitoring, weather prediction, and industrial process control. Accurate reconstruction of sensor data and reliable forecasting are essential for making informed decisions and ensuring the smooth operation of various systems. In this homework assignment, I conducted a comprehensive analysis of the performance of a sensor reconstruction and forecasting model.

The primary objective of this analysis was to investigate the impact of several key factors on the model's performance. Specifically, I focused on three factors: time lag, sensor variation, and noise. Understanding how these factors influence the model's accuracy and robustness is essential for optimizing its performance in real-world applications.

To evaluate the impact of time lag, I varied the time lag parameter and examined how different time lags affected the accuracy of the reconstructed sensor data. By considering different time lags, I aimed to understand the relationship between the temporal context provided to the model and its ability to reconstruct and forecast sensor data effectively.

Furthermore, I explored the influence of sensor variation on the model's performance. By varying the number of sensors, I assessed the model's capability to handle different sensor configurations. This analysis allowed me to understand the model's adaptability to diverse sensor layouts and the trade-offs between the number of sensors and the reconstruction accuracy.

Finally, I investigated the effect of noise on the model's performance. By introducing Gaussian noise to the sensor data, I examined how different noise levels impacted the accuracy of the reconstructed sensor data. This analysis provided insights into the model's robustness and its ability to handle noisy real-world sensor measurements.

Through these comprehensive analyses, I aimed to gain a deeper understanding of the model's behavior and its performance under various conditions. The insights gained from this study can guide the optimization of sensor reconstruction and forecasting models, leading to improved decision-making and system performance in practical applications.
## Theoretical Background
Sensor reconstruction and forecasting models are essential tools for analyzing and predicting sensor data in various domains. These models leverage techniques from time series analysis, machine learning, and signal processing to reconstruct missing or corrupted sensor measurements and forecast future values. In this section, I provide a brief overview of the key concepts and methodologies underlying the sensor reconstruction and forecasting task.

- Time Lag:
The time lag parameter refers to the temporal context provided to the model for reconstructing and forecasting sensor data. By considering historical sensor measurements within a specific time window, the model can capture temporal dependencies and patterns in the data. Different time lags can have varying effects on the model's accuracy, as a shorter time lag might capture more immediate changes, while a longer time lag might capture more long-term trends. By analyzing the performance of the model across different time lags, I can understand the optimal temporal context for accurate sensor reconstruction and forecasting.

- Sensor Variation:
Sensor variation refers to the diversity in the number and placement of sensors used for data collection. In real-world scenarios, sensor networks can have different configurations, including variations in the number of sensors and their spatial distribution. The performance of a sensor reconstruction and forecasting model can be influenced by the number of sensors employed and their placement. A model that can effectively handle different sensor variations would demonstrate adaptability and scalability across different sensor layouts. By analyzing the model's performance under varying sensor configurations, I can assess its robustness and scalability.

- Noise:
Noise is an inherent component of sensor measurements and can arise from various sources, including measurement errors, environmental factors, or communication artifacts. The presence of noise in sensor data can significantly impact the accuracy of reconstruction and forecasting models. Therefore, understanding the behavior of the model under different noise levels is crucial. Gaussian noise is a commonly used model for simulating random measurement errors. By introducing Gaussian noise to the sensor data at different levels, we can analyze the model's performance and its ability to handle noisy sensor measurements. Evaluating the model's robustness to noise can provide insights into its reliability and suitability for real-world applications.

In this homework assignment, I aim to analyze the performance of a sensor reconstruction and forecasting model by varying the time lag, sensor variation, and noise levels. By examining these factors, I can gain a deeper understanding of the model's behavior and its performance under different conditions. This theoretical background sets the foundation for the subsequent analysis and interpretation of the experimental results.
## Algorithm Implementation
### Data Preparation:
Load the sensor data using the provided load_data function.
Define the number of sensors (num_sensors) and the desired time lag (lags).
Split the data into training, validation, and test sets.
```
load_X = load_data('SST')
n = load_X.shape[0]
m = load_X.shape[1]
sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
train_indices = np.random.choice(n - lags, size=1000, replace=False)
mask = np.ones(n - lags)
mask[train_indices] = 0
valid_test_indices = np.arange(0, n - lags)[np.where(mask!=0)[0]]
valid_indices = valid_test_indices[::2]
test_indices = valid_test_indices[1::2]
```
### Preprocessing:
Apply feature scaling to the sensor data using the MinMaxScaler from sklearn.preprocessing.
Fit the scaler on the training data and transform the entire dataset.
Generate input sequences for the SHRED model by selecting sensor measurements based on the sensor locations and the time lag.
Create training, validation, and test datasets for both sensor reconstruction and forecasting tasks.
```
sc = MinMaxScaler()
sc = sc.fit(load_X[train_indices])
transformed_X = sc.transform(load_X)

### Generate input sequences to a SHRED model
all_data_in = np.zeros((n - lags, lags, num_sensors))
for i in range(len(all_data_in)):
    all_data_in[i] = transformed_X[i:i+lags, sensor_locations]

### Generate training validation and test datasets both for reconstruction of states and forecasting sensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data_in = torch.tensor(all_data_in[train_indices], dtype=torch.float32).to(device)
valid_data_in = torch.tensor(all_data_in[valid_indices], dtype=torch.float32).to(device)
test_data_in = torch.tensor(all_data_in[test_indices], dtype=torch.float32).to(device)

### -1 to have output be at the same time as final sensor measurements
train_data_out = torch.tensor(transformed_X[train_indices + lags - 1], dtype=torch.float32).to(device)
valid_data_out = torch.tensor(transformed_X[valid_indices + lags - 1], dtype=torch.float32).to(device)
test_data_out = torch.tensor(transformed_X[test_indices + lags - 1], dtype=torch.float32).to(device)

train_dataset = TimeSeriesDataset(train_data_in, train_data_out)
valid_dataset = TimeSeriesDataset(valid_data_in, valid_data_out)
test_dataset = TimeSeriesDataset(test_data_in, test_data_out)
```
### Model Training:
Initialize an instance of the SHRED model with the desired parameters, such as the number of sensors, hidden size, and number of hidden layers.
Move the model to the available device (e.g., GPU).
Use the fit function from the models module to train the SHRED model.
Provide the training and validation datasets, along with the desired batch size, number of epochs, learning rate, and early stopping patience.
Monitor the validation errors during training.
```
shred = models.SHRED(num_sensors, m, hidden_size=64, hidden_layers=2, l1=350, l2=400, dropout=0.1).to(device)
validation_errors = models.fit(shred, train_dataset, valid_dataset, batch_size=64, num_epochs=1000, lr=1e-3, verbose=True, patience=5)
```
### Evaluation:
Apply the trained model to reconstruct sensor data for the test dataset.
Inverse transform the reconstructed sensor data and the ground truth data using the scaler.
Compute the performance metric, such as the reconstruction error, by comparing the reconstructed data with the ground truth data.
Visualize the reconstructed and ground truth data for qualitative analysis.
```
test_recons = sc.inverse_transform(shred(test_dataset.X).detach().cpu().numpy())
test_ground_truth = sc.inverse_transform(test_dataset.Y.detach().cpu().numpy())
print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
```
### Performance Analysis:
Perform additional analyses to evaluate the impact of various factors on the model's performance:

- Time Lag: Vary the time lag parameter and observe its effect on the reconstruction accuracy.
```
time_lag_values = [10, 20, 30, 40, 50] 

performance_metrics = []  # List to store the performance metrics for each time lag

for lags in time_lag_values:
```
- Noise: Introduce Gaussian noise to the sensor data at different levels and analyze its impact on the reconstruction accuracy.
```
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4] 

performance_metrics = []  # List to store the performance metrics for each noise level

for noise_level in noise_levels:
    # Add Gaussian noise to the data
    noisy_load_X = load_X + np.random.normal(loc=0, scale=noise_level, size=load_X.shape)
```
- Sensor Variation: Vary the number of sensors and assess the model's ability to handle different sensor configurations.
```
num_sensor_values = [2, 4, 6, 8, 10] 

performance_metrics = []  # List to store the performance metrics for each number of sensors

for num_sensors in num_sensor_values:
    # Update the number of sensors in the code
    sensor_locations = np.random.choice(m, size=num_sensors, replace=False)
```
### Analyze and Interpret Results:
Analyze the performance metrics obtained from different experiments.
```
# Plot the performance as a function of the number of sensors variable
plt.plot(num_sensor_values, performance_metrics)
plt.xlabel('Number of Sensors')
plt.ylabel('Reconstruction Error')
plt.title('Performance as a Function of Number of Sensors')
plt.show()
```
## Computational Results
### Example train and plot
Using the given parameters of 3 sensors, a time lag of 52, and no added noise, the trained model achieved a mean square error of 0.019370167 between the reconstructed weekly mean sea-surface temperature and the ground truth. A comparison between one of the sampled reconstructed outputs and the ground truth is depicted in the accompanying figure.
![image](https://github.com/samtam1118/EE399/assets/110360994/6a81cb36-dcc3-4508-a523-576a022991ff)
### Time Lag Analysis
I varied the time lag parameter in the sensor reconstruction and forecasting model and evaluated its impact on the performance. The following results were obtained:

![image](https://github.com/samtam1118/EE399/assets/110360994/c628c51b-221a-4597-89af-1d24dbc42f95)

The results indicate that as the time lag increases, the reconstruction error decreases. This suggests that a longer time lag provides a better temporal context for the model to capture relevant patterns and improve the accuracy of sensor reconstruction and forecasting.

### Sensor Variation Analysis
I investigated the model's performance by varying the number of sensors in the sensor network. The following results were obtained:

![image](https://github.com/samtam1118/EE399/assets/110360994/249a2ef1-6333-4c43-aba7-aa1ea69a2a80)

The results show that increasing the number of sensors improves the reconstruction accuracy. This indicates that a larger number of sensors provides more comprehensive coverage of the spatial domain and enhances the model's ability to reconstruct sensor measurements accurately.

### Noise Analysis
I examined the impact of Gaussian noise on the model's performance by introducing noise at different levels into the sensor data. The following results were obtained:

![image](https://github.com/samtam1118/EE399/assets/110360994/1d997540-699e-44c2-945a-f2e4825e5290)

The results demonstrate that as the noise level increases, the reconstruction error also increases. This highlights the sensitivity of the model to noisy sensor measurements and emphasizes the importance of noise reduction techniques for accurate sensor reconstruction and forecasting.

## Summary and Conclusions
In this homework assignment, I trained an LSTM/decoder model for sea-surface temperature (SST) data. The model aimed to reconstruct the SST states and forecast sensor measurements based on historical data. I performed several analyses to evaluate the model's performance under different conditions.

First, I trained the model using a specified number of sensors and a time lag variable. The training process involved splitting the data into training, validation, and test sets, normalizing the data, and creating input sequences for the model. The LSTM/decoder model was trained using the training set and evaluated on the validation set.

Next, I plotted the results by comparing the reconstructed SST states with the ground truth SST data. I used the inverse transformation to obtain the reconstructed data in the original scale. The plots provided a visual representation of the model's performance in capturing the underlying patterns and variations in the SST data.

Furthermore, I conducted an analysis of the model's performance as a function of the time lag variable. By varying the time lag, I observed changes in the accuracy of the reconstructed SST states and the forecasting of sensor measurements. This analysis helped me understand the optimal time lag for capturing meaningful temporal dependencies in the data.

I also analyzed the model's performance as a function of noise by adding Gaussian noise to the SST data. By evaluating the reconstructed data under different noise levels, I assessed the model's robustness and ability to handle noisy inputs. This analysis provided insights into the model's performance in real-world scenarios where data may contain inherent noise.

Lastly, I investigated the performance of the model as a function of the number of sensors. By varying the number of sensors used in the input sequences, I evaluated the model's capability to handle different levels of spatial information. This analysis shed light on the trade-off between the number of sensors and the accuracy of the reconstructed SST states and sensor forecasts.

Overall, the LSTM/decoder model demonstrated promising results in reconstructing SST states and forecasting sensor measurements. It showed sensitivity to the time lag variable, noise levels, and the number of sensors. These findings contribute to our understanding of the model's capabilities and limitations in capturing and predicting spatiotemporal patterns in SST data.
