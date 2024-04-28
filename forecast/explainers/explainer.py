from abc import ABCMeta, abstractmethod
from scipy.special import comb
import math
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class BlackBox:
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, X):
        return

    @abstractmethod
    def predict_proba(self, X):
        return
    
    @abstractmethod
    def predict_2Dto3D(self, X):
        return
    
class Explainer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit_exp(self, model, data, features_name):
        return
    
    @abstractmethod
    def shap_values_multivariate(self, X):
        return
    
    @abstractmethod
    def shap_multivariate(self, X):
        return


class BlackBoxWrapper(BlackBox):
    def __init__(self, model):
        self.clf = model

    def predict(self, X):
        y = self.clf.predict(X)
        return y
    
    def predict_proba(self, X):
        y = self.clf.predict_proba(X)
        return y
    
    def predict_2Dto3D(self, X):
        y = transform_to_3d(X)
        x = self.predict(y)
        return x
    
## 
#   Data wrapper - functions
##
    
def is_3d(data):
    y = False
    if data.ndim == 3:
        y = True
    return y

def is_2d(data):
    y = False
    if data.ndim == 2:
        y = True
    return y

def transform_to_3d(data):
    y = data.reshape((1, data.shape[0], data.shape[1])) #(num_samples, sample_size, nu_features)
    return y

def transform_to_2d(data):
    y = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
    return y


class ITPFExplainer(Explainer):
    '''
    The Explainer class.

    Parameters:
    model: model or prediction function of model.
    data_x: Data used to initialize the explainers with, (3D shape).
    data_y: Data used to explain the model, (2D shape).
    features-names: List of features names.
    class-names: List of the class name to be explained.
    '''
    def __init__(self):
        self.model = None
        self.data_x = None
        self.data_y = None
        self.feature_names = None
        self.class_name = None
        self.feature_num = None
        self.feature_pred_num = None

    def fit_exp(self, model, x, y, feature_names, class_names, feature_nr, feature_pred_nr):
        self.model = model
        self.data_x = x
        self.data_y = y
        self.feature_names = feature_names
        self.class_names = class_names
        self.feature_num = feature_nr
        self.feature_pred_num = feature_pred_nr
    

    def _generate_perturbed_samples(self, original_sequence, scale=3.5,):
        """
        Generates perturbed samples that are significantly different from the original sequence.
        
        Parameters:
            original_sequence (np.ndarray): The original sequence of shape (timesteps, features).
            scale (float): Scale of perturbation. For 'range_based', it's a percentage of the feature range.
                        For 'fixed_noise', it's the magnitude of noise added directly.
        
        Returns:
            np.ndarray: A single perturbed sample of the same shape as the original sequence.
        """
        # Ensure the input is a NumPy array for manipulation
        original_sequence = np.array(original_sequence)

        # Initialize an array to hold the perturbed samples
        perturbed_samples = np.zeros((original_sequence.shape[0], original_sequence.shape[1]))

        # Calculate the standard deviation of the original sequence
        std_devs = np.std(original_sequence, axis=0)

        # Generate noise
        perturbation = (std_devs + scale) * (original_sequence[1])
        
        # Generate the perturbed sample
        perturbed_samples = original_sequence + perturbation
        
        return perturbed_samples
    
    def _perturb_time_series(self, matrix):

        # Convert the 2D array to a DataFrame
        df = pd.DataFrame(matrix)
        #compute the correlation DataFrame
        corr_DataFrame = df.corr()
        # Convert DataFrame to NumPy array
        corr_matrix = corr_DataFrame.values
        # Replace NaN values with 0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0)
        
        # Initialize perturbation matrix
        perturbed_matrix = np.copy(matrix)
        
        # Iterate over each feature
        for i in range(matrix.shape[1]):
            # Find indices of features with high correlation with feature i
            correlated_features = np.where(np.abs(corr_matrix[i]) > 0.7)[0]
            
            # Compute perturbation values based on correlation grade
            for j in correlated_features:
                if i != j:
                    perturbation_factor = corr_matrix[i, j]
                    perturbed_matrix[:, j] += perturbation_factor * perturbed_matrix[:, i]
        
        # Add noise perturbation for features with low correlation
        for i in range(matrix.shape[1]):
            if np.abs(corr_matrix[i]).max() <= 0.7:
                perturbed_matrix[:, i] *= (100 + (np.mean(perturbed_matrix[:, i]) * ((matrix.shape[1])/2) * (np.std(perturbed_matrix[:, i]))))
        
        return perturbed_matrix

    
    def _baseline(self, y):
        pertubed_samples = self._perturb_time_series(y)

        return pertubed_samples
    
        
    def _compute_weight(self, total_features, subset_size):
        """
        Computes the weight for subsets of a given size in the context of Shapley values.
        
        Parameters:
        - total_features: Total number of features (|N|).
        - subset_size: Size of the subset including the feature (|S|).
        
        Returns:
        - The weight for the subset.
        """
        return (math.factorial(subset_size - 1) * math.factorial(total_features - subset_size)) / math.factorial(total_features)
   
    
    def _create_mask(self, feature_num):
        mask = np.array(list(product(range(2), repeat=feature_num)))
        mask = mask[~np.all(mask == 0, axis=1)]
        mask = mask[~np.all(mask == 1, axis=1)]
    
        return mask
    
    
    def _value_function(self, feature, feature_num, y, pertubed_data, model):

        # create coalition matrix for masking
        mask=self._create_mask(feature_num)

        # configuration
        marginal_contribution = []
        average_contribution = []
        data = y

        # baseline or pertubed data
        baseline_tab = pertubed_data
        
        # compute baseline prediction
        baseline_pred = model.predict_2Dto3D(baseline_tab)
        # compute y_true
        original_pred = model.predict_2Dto3D(data)

        # reshap prediction 3d to 2d
        baseline_pred_reshap = transform_to_2d(baseline_pred)
        original_pred_reshap = transform_to_2d(original_pred)

        # compute RMSEvalue original_pred vs baseline_pres
        rmse = mean_squared_error(original_pred_reshap, baseline_pred_reshap)

        # compute marginal contribution for baseline {feature} - {}
        weight = self._compute_weight(feature_num, 1)

        pred = weight*(rmse)
        marginal_contribution.append(pred)

        # compute marginal contribution for the rest of the combinations
        for item in mask:
            # Initialize arrays as copies of data and baseline_tab
            without_feature = baseline_tab.copy()
            with_feature = baseline_tab.copy()
            number_of_feature_masked = 0
            
            # Iterate over each element in the mask
            for i, use_data in enumerate(item):
                if use_data:
                    without_feature[:, i] = data[:, i]
                    with_feature[:, i] = data[:, i]
                    number_of_feature_masked = number_of_feature_masked + 1

            # Include the feature
            with_feature[:, feature] = data[:, feature]
            number_of_feature_masked = number_of_feature_masked + 1
            weight = self._compute_weight(feature_num, number_of_feature_masked)
        

            # compute marginal contribution
            pred_without_feature = model.predict_2Dto3D(without_feature)
            pred_with_feature = model.predict_2Dto3D(with_feature)

            # reshap prediction 3d to 2d
            pred_without_feature_reshap = transform_to_2d(pred_without_feature)
            pred_with_feature_reshap = transform_to_2d(pred_with_feature)

            # compute rmse without_feature vs original_pred, rmse with_feature vs original_pred
            rmse_without_feature = mean_squared_error(original_pred_reshap, pred_without_feature_reshap)
            rmse_with_feature = mean_squared_error(original_pred_reshap, pred_with_feature_reshap)

            pred = weight*(rmse_with_feature - rmse_without_feature)
            marginal_contribution.append(pred)
        
        # compute the average contribution
        average_contribution = np.mean(marginal_contribution)
        return average_contribution
    
    def shap_values_multivariate(self, y):
        # step 1: configuration
        pertubed_data = self._perturb_time_series(y)
        n_features = self.feature_num
        shapley_values = []
        
        # step 2: Compute the shapeley values for each feature
        for feature in range(n_features):
            s_value = self._value_function(feature, n_features, y, pertubed_data, self.model)
            # Load values
            shapley_values.append(s_value)
        return shapley_values, pertubed_data
    
    def shap_multivariate(self, s_values):
        shap_values = s_values
        feature_names = self.feature_names

        # Determine bar colors based on the sign of the mean Shapley values
        bar_colors = ['red' if x < 0 else 'blue' for x in shap_values]

        # Plot
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(feature_names))
        plt.barh(y_pos, shap_values, align='center', color=bar_colors)
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Mean SHAP Value')
        plt.title('Feature Importance with SHAP Values')
        plt.show()

## Gaussian process
## Prediction uncertainity
## Prediction interval
    
## Contribution of individual covariate:
    # - Shapley value

## Partial Dependence - Link function shape
    
#def predict(self, X):
    # here the input is 2-d and we want to transform it to 3-d before prediction
#    x = X.reshape((X.shape[0], X.shape[1], 1))
#    y = self.clf.predict(x)
#    return y