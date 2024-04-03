from abc import ABCMeta, abstractmethod
from sklearn.metrics import accuracy_score
import numpy as np
from lime import lime_tabular
import shap

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
    y = data.reshape((data.shape[0], data.shape[1], 1)) #(num_samples, sample_size, nu_features)
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

    def fit_exp(self, model, x, y, feature_names, class_names):
        self.model = model
        self.data_x = x
        self.data_y = y
        self.feature_names = feature_names
        self.class_names = class_names
    
    def lime(self, y, labelId):
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