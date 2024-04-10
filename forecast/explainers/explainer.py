from abc import ABCMeta, abstractmethod
from sklearn.metrics import accuracy_score
import numpy as np
from lime import lime_tabular
import shap
from scipy.special import comb
import math
import numpy as np
import itertools
import sys
from itertools import product
import matplotlib.pyplot as plt


class BlackBox:
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, X):
        return

    @abstractmethod
    def predict_proba(self, X):
        return
    
    @abstractmethod
    def predict_for_lime(self, X):
        return
    
    @abstractmethod
    def predict_for_shap(self, X):
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
    def lime(self, X):
        return
    
    @abstractmethod
    def shap(self, X):
        return
    
    @abstractmethod
    def shap_values_multivariate(self, X):
        return
    
    @abstractmethod
    def shap_multivariate(self, X):
        return


class BlackBoxWrapper(BlackBox):
    def __init__(self, model, isMultivariate=False):
        self.clf = model
        self.isMultivariate = isMultivariate

    def predict(self, X):
        y = self.clf.predict(X)
        return y
    
    def predict_proba(self, X):
        y = self.clf.predict_proba(X)
        return y
    
    def predict_for_lime(self, X):
        print(X.shape) # Check Lime synthetic data around X
        y = self.predict(X)
        return y
    
    def predict_for_shap(self, X):
        #y = transform_to_3d(X)
        y = X.reshape((X.shape[0], X.shape[1], 1))
        #print(y)
        x = self.predict(y)
        return x
    
    def predict_2Dto3D(self, X):
        y = transform_to_3d(X)
        print(y.shape)
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

##
# Global variables
#
PRED_SIZE = 1

class ITPFExplainer(Explainer):
    '''
    The Explainer class.

    Parameters:
    model: model or prediction function of model.
    data_x: Data used to initialize the explainers with. 3D shape.
    data_y: Data used for explain the model. 2D shape.
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
    

    def _generate_significantly_perturbed_samples(self, original_sequence, scale=3.5,):
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
        perturbation = (std_devs + scale) * (original_sequence[1]) * np.random.randn(*original_sequence.shape)
        
        # Generate the perturbed sample
        perturbed_samples = original_sequence + perturbation
        
        return perturbed_samples


    
    def _generate_pertubed_samples(self, original_sequence, n_samples=0.05, noise_level=100.0):
        """
        Generates a dataset of perturbed samples around the original time series data point.
        
        Parameters:
            original_sequence (np.ndarray): The original sequence of shape (timesteps, features),
                                            e.g., (59, 8).
            n_samples (int): Number of perturbed samples to generate.
            noise_level (float): Magnitude of noise to add for perturbation, relative to the
                                standard deviation of each feature.
        
        Returns:
            np.ndarray: A dataset of perturbed samples of shape (n_samples, timesteps, features).
        """
        # Ensure the input is a NumPy array for manipulation
        original_sequence = np.array(original_sequence)
        
        # Calculate the standard deviation of the original sequence
        std_devs = np.std(original_sequence, axis=0)
        
        # Initialize an array to hold the perturbed samples
        perturbed_samples = np.zeros((n_samples, *original_sequence.shape))
        
        # Generate perturbed samples
        for i in range(n_samples):
            # Generate random noise
            noise = np.random.randn(*original_sequence.shape) * std_devs * noise_level
            
            # Add noise to the original sequence to create a perturbed sample
            perturbed_samples[i] = original_sequence + noise
        
        return perturbed_samples
    
    def _baseline(self, y):
        pertubed_samples = self._generate_pertubed_samples(y)
        mean_pertubed_samples = []
        mean_pertubed_samples_pred = []
        
        # predict on the new sample
        p = self.model.predict(pertubed_samples)

        # Get the mean of the samples and their predictions
        mean_pertubed_samples = np.mean(pertubed_samples, axis=0)
        mean_pertubed_samples_pred = np.mean(p, axis=0)

        return mean_pertubed_samples, mean_pertubed_samples_pred
    
    def _baseline_extra(self, y):
        pertubed_samples = self._generate_significantly_perturbed_samples(y)
        return pertubed_samples


    
    def lime(self, y, labelId=0):
        ''' 
        LIME - Lime-tabular - RecurrentTabularExplainer
        An explainer for keras-style recurrent neural networks, where the input shape is (n_samples, n_timesteps, n_features). 
        This class just extends the LimeTabularExplainer class and reshapes the training data and feature names such that they become something like
        (val1_t1, val1_t2, val1_t3, …, val2_t1, …, valn_tn)

        Parameters:
        y: The data for getting interpretability. A numpy 2D-array (sample-size, features-number).
        '''
        # Define the explainer
        explainer = lime_tabular.RecurrentTabularExplainer(
            training_data=self.data_x,
            training_labels=self.data_y,
            feature_names=self.feature_names,
            discretize_continuous=True,
            class_names=self.class_names,
            discretizer='decile'
        )
        # Get the the result
        exp = explainer.explain_instance(y, self.model.predict_for_lime, num_features=10, labels=(labelId,))
        exp.show_in_notebook()
        
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
   

    def shap(self, y):
        '''
        Uses the Kernel SHAP method to explain output of any function.
        Takes a matrix of samples (# samples x # features) and computes 
        the output of the model for those samples.

        Parameters:
        data: The background dataset to use for integrating out features. A numpy 2D-array (sample-size, features-number).
        y: The data for getting interpretability. A numpy 2D-array (sample-size, features-number).
        '''
        shap.initjs()
        # Define the explainer
        explainer = shap.KernelExplainer(model=self.model.predict_for_shap, data=self.data_x[0], feature_names=self.feature_names)
        # Get shap values
        shap_values = explainer.shap_values(y)
        shap.summary_plot(shap_values, y)
    
    def _create_mask(self, feature_num):
        mask = np.array(list(product(range(2), repeat=feature_num)))
        mask = mask[~np.all(mask == 0, axis=1)]
        mask = mask[~np.all(mask == 1, axis=1)]
    
        return mask
    
    def _value_function(self, feature, feature_num, y, model):
        # create array mask
        mask=self._create_mask(feature_num)

        # configuration and baseline
        marginal_contribution = []
        average_contribution = []
        data = y
        baseline_tab = self._baseline_extra(data)
        n_features = data.shape[1]
        baseline_pred = model.predict_2Dto3D(baseline_tab)
        original_pred = model.predict_2Dto3D(data)

        # compute marginal contribution for baseline {feature} - {}
        weight = self._compute_weight(feature_num, 1)
        pred = weight*(original_pred - baseline_pred)
        marginal_contribution.append(pred)
        # compute marginal contribution

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
            pred = weight*(pred_with_feature - pred_without_feature)
            marginal_contribution.append(pred)
        
        # compute the average contribution
        marginal_contributions1_reshape = np.array([a.reshape(1, -1) for a in marginal_contribution])
        marginal_contribution_combined = np.vstack(marginal_contributions1_reshape)
        average_contribution = np.mean(marginal_contribution_combined, axis=0)
        return average_contribution
    
    def shap_values_multivariate(self, y):
        # step 1: Get the baseline and configuration
        n_features = self.feature_num
        n_feature_pred = self.feature_pred_num
        shapley_values = np.zeros((n_feature_pred, n_features))
        # step 2: Compute the shapeley values for each feature
        for feature in range(n_features):
            s_value = self._value_function(feature, n_features, y, self.model)
            #print("Feature ", feature, " values: ", s_value)
            # Load values
            for p in range(n_feature_pred):
                shapley_values[p][feature] = s_value[p]
        return shapley_values
    
    def shap_multivariate(self, s_values, labelId):
        shap_values = s_values[labelId]  # Your actual Shapley values
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