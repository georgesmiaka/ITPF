# Vehicle Trajectory Prediction Framework (VTPF)

## Overview

**Vehicle Trajectory Prediction Framework (ITPF)** is a research-driven project aimed at predicting the next location of a driver based on their driving history. The focus is to develop models that can accurately forecast future locations using multivariate time series data. The framework tackles critical challenges in car trajectory prediction, such as model selection, data scarcity, validation of simulated data, and providing interpretability for model decisions.

## Features & Input Data

ITPF relies on **multivariate time series data**, which includes multiple variables evolving over time. The main features used for prediction are:

- **x, y, z**: Spatial coordinates of the vehicle
- **Heading**: The direction in which the vehicle is facing
- **Velocity**: The speed of the vehicle
- **Weather**: Environmental conditions such as rain, snow, or clear skies
- **Trip ID**: Identifier for each trip to distinguish multiple driving sessions
- **Time Index (seconds)**: A timestamp marking the position of each data point
- **Date (year-month-day hour-minute)**: Full date and time to capture seasonal, daily, and time-of-day variations

The output of the framework is the **next series of localizations** (next x, y, z coordinates), allowing for trajectory prediction across time.

## Project Goals

1. **Predict Next Locations**: Develop models that consider the driving history and predict the next spatial coordinates (x, y, z) of the vehicle.
   
2. **Overcome Data Scarcity**: Due to privacy and trust concerns, real-world trajectory data can be difficult to obtain. To address this, we use **Scenic** and **CARLA**, which are tools for simulating driving environments, to generate synthetic data for model training and testing.

3. **Validate Simulation vs. Reality**: A key challenge is to ensure the validity of simulated data. We explore methods to compare and validate the simulated data against real-world driving data.

4. **Interpretability of Predictions**: It is important not only to make accurate predictions but also to understand and explain the decision-making process of the models. This involves the selection and implementation of interpretable machine learning techniques to clarify why certain predictions were made.

## Challenges

1. **Multivariate Time Series Data**: 
   - In trajectory prediction, multiple variables (such as position, velocity, and environmental factors) change over time. Handling this type of data is complex, as it requires capturing temporal relationships between these variables.
   - **Multivariate time series modeling** involves predicting multiple future steps of several interrelated variables simultaneously. These models must account for both the temporal dependencies within each variable and the relationships between different variables over time.

2. **Data Scarcity**:
   - Access to real-world driving data is often restricted due to privacy concerns, making it difficult to gather sufficient data for training robust models. 
   - To overcome this, **Scenic** (a scenario specification language) and **CARLA** (an autonomous driving simulator) are used to generate realistic, diverse driving scenarios and produce synthetic trajectory data.

3. **Simulation-to-Reality Validation**:
   - A significant obstacle is validating the performance of models trained on simulated data in real-world conditions. We are exploring various validation techniques to ensure the simulated environments are realistic enough for reliable prediction.
   - Methods include comparing key metrics and patterns from simulated and real datasets to assess the generalization capabilities of the model.

4. **Interpretability**:
   - One of the core aspects of this project is the interpretability of predictions. Understanding **why** a model makes a certain prediction helps in gaining trust and improving the model.
   - Techniques such as **SHAP (Shapley Additive Explanations)** or **LIME (Local Interpretable Model-agnostic Explanations)** could be applied to explain the influence of each feature (e.g., weather, speed, heading) on the final prediction.

## Framework Components

- **Data Generation with Scenic and CARLA**: Efficiently produce car trajectory data for training and testing models, mimicking real-world driving scenarios.
  
- **Model Selection for Trajectory Prediction**: Explore and implement models suitable for predicting trajectories in multivariate time series data. This may include classical models like **ARIMA**, **LSTM (Long Short-Term Memory)**, and **GRU (Gated Recurrent Unit)**, as well as newer methods such as **Transformers**.

- **Sim-to-Real Validation Methods**: Implement methods to validate the consistency and accuracy of the simulated data against real-world driving data.

- **Interpretability Tools**: Implement interpretability models that provide insights into the predictions made by the framework. These tools will help explain the model’s decision-making process to users and stakeholders.

## Conclusion

The **Interpretable Trajectory Prediction Framework (ITPF)** aims to provide an effective, interpretable solution for predicting a driver’s next location using multivariate time series data. By leveraging simulation tools for data generation, selecting appropriate predictive models, and focusing on interpretability, the framework seeks to contribute to research in trajectory prediction while addressing the challenges of data scarcity and validation.
