from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from torch import tensor, from_numpy
warnings.filterwarnings('ignore')

class IMT():
    '''
    Interpret Multivariate Timeseries,
    A class for displaying various interpretability related metrics. 
    
    Parameters:
    model: A model object that will be used for prediction. The model object must have a method called predict() which produces the model output for a given input
    example_x: Example of the structure and size of an input to the model
    example_y: Example of the structure and size of an output of the model
    features: list of strings with all the features in the input.
    ignore: list of feature indices in the input to ignore
    '''
    def __init__(self, predict_fn, loss, example_x, example_y, features, ignore = []):
        self.predict_fn = predict_fn
        self.loss = loss
        self.example_x = example_x
        self.example_y = example_y
        self.features = features
        self.ignore = ignore

        # Data related sizes,
        self.timesteps_in = example_x.shape[0] 
        self.timesteps_out = example_y.shape[0]
        self.features_in = example_x.shape[1]
        self.target_features = example_y.shape[1]

        self.features_in_no_ignores = features
        ignore.sort(reverse = True)
        for i in ignore:
            del self.features_in_no_ignores[i]

        print(self.features_in_no_ignores)

    def avg_feature_importance(self, x, target = None):
        '''
        The average feature importance, computed by masking all the features (one-by-one) and comparing the masked prediction to the ground truth.

        x: input to the model on the form (timesteps_in, features_in)
        target: target ouput, if none we default to the predict_fn predictions of x
        '''        
        if target is None:
            target = self.predict_fn(x)
        losses = np.zeros(shape=(len(self.features_in_no_ignores)))
        masks = []
        mask = np.ones(shape=(self.features_in), dtype=bool)

        for i in range(0,self.features_in):
            if i in self.ignore:
                continue
            mask_feature = mask.copy()
            mask_feature[i] = False
            masks.append(mask_feature)

        for i, mask in enumerate(masks):
            f = lambda x_row: x_row * mask
            masked_x = f(x)
            y_masked = self.predict_fn(masked_x)
            loss = self.loss(y_masked, target)
            losses[i] = loss
        
        fig, ax = plt.subplots(figsize = (8.8,8))
        order = np.argsort(losses)
        ax.barh(np.arange(len(losses)), (losses[order] / sum(losses)) * 100, tick_label=np.asarray(self.features_in_no_ignores)[order])
        ax.set_title("Average Feature Importance")
        ax.set_xlabel("Importance in %")
        plt.plot()
        return

    def timestep_importance(self, x, target = None):
        '''
        The timestep importance, computed by masking all the timesteps (one-by-one) and comparing the prediction to the ground truth.

        x: input to the model on the form (timesteps_in, features_in)
        target: target ouput, if none we default to the predict_fn predictions of x
        '''
        if target is None:
            target = self.predict_fn(x)
        
        losses = np.zeros(shape=(self.timesteps_in))
        masks = []
        mask = np.ones(shape=(self.timesteps_in, self.features_in), dtype=bool)

        for i in range(0,self.timesteps_in):
            mask_ts = mask.copy()
            ignores = np.zeros(shape=(self.features_in), dtype=bool)
            for n in self.ignore:
                ignores[n] = True
            mask_ts[i] = ignores
            masks.append(mask_ts)

        for i, mask in enumerate(masks):
            masked_x = mask*x
            y_masked = self.predict_fn(masked_x)
            loss = self.loss(y_masked, target)
            losses[i] = loss

        fig, ax = plt.subplots(figsize = (9.2,8))
        ax.plot(range(-1 * self.timesteps_in, 0), losses)
        ax.set_title("Timestep importance")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Importance")
        plt.plot()
        return
    
    def windowed_feature_importance(self, x, target = None, window_size = 10, stride = 10):
        '''
        Averaged feature importance across the sliding window

        x: input to the model on the form (timesteps_in, features_in)
        window_size: size of the sliding window
        stride: how much the window moves every itteration
        '''
        if target is None:
            target = self.predict_fn(x)
        
        mask = np.ones(shape=(self.timesteps_in, self.features_in), dtype=bool)
        window_starts = [i for i in range(0,self.timesteps_in,stride) if i+window_size <= self.timesteps_in]
        losses = np.zeros(shape=(len(window_starts), len(self.features_in_no_ignores)))

        for i,start in enumerate(window_starts):
            masks = []
            for j in range(0,self.features_in):
                if j in self.ignore:
                    continue
                masked_window_feature = mask.copy()
                masked_window_feature[start:(window_size+start),j] = np.zeros(shape=(window_size), dtype=bool)
                masks.append(masked_window_feature)
            for feature,wf_mask in enumerate(masks):
                masked_x = wf_mask*x
                y_masked = self.predict_fn(masked_x)
                loss = self.loss(y_masked, target)
                losses[i][feature] = loss

        windows = [str((i,i+window_size)) for i in window_starts]
        plot_ready = pd.DataFrame(losses, columns=self.features_in_no_ignores)
        plot_ready = plot_ready.astype(float)
        plot_ready.loc[:, "window"] = windows
        plot_ready.plot(x='window', kind='bar', stacked=True, title='Windowed Feature Importance', figsize=(9.3,8))
        plt.show()
        return
    
    def feature_importance(self, x, target = None):
        '''
        The importance of every feature at every timestep, creates a lineplot for all the features.
        Very computational expensive, think masking every single feature at every single timestep.

        x: input to the model on the form (timesteps_in, features_in)
        '''
        if target is None:
            target = self.predict_fn(x)
            
        mask = np.ones(shape=(self.timesteps_in, self.features_in), dtype=bool)
        losses = np.zeros(shape=(self.timesteps_in, self.features_in))

        for i in range(0,self.timesteps_in):
            for j in range(0,self.features_in):
                if j in self.ignore:
                    continue
                masked_feature_timestep = mask.copy()
                masked_feature_timestep[i][j] = False
                masked_x = masked_feature_timestep * x
                y_masked = self.predict_fn(masked_x)
                loss = self.loss(y_masked, target)
                losses[i][j] = loss
        fig, ax = plt.subplots(figsize = (9.3,8))
        
        for feature in (losses.transpose()):
            ax.plot(range(-1 * self.timesteps_in, 0), feature)
        ax.set_title("Feature Importance Across All Timesteps")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Importance")
        ax.legend(self.features)
        plt.plot()
        return
    
    def output_input_importance(self, x, target = None):
        '''
        How masking out inputs relate to specific timesteps in the future, maybe instead we should do some kind of grouping or selecting the 5-10 most important timesteps for every output timestep

        x: input to the model on the form (timesteps_in, features_in)
        '''
        if target is None:
            target = self.predict_fn(x)
        
        losses = np.zeros(shape=(self.timesteps_in, self.timesteps_out))
        masks = []
        mask = np.ones(shape=(self.timesteps_in, self.features_in), dtype=bool)

        for i in range(0,self.timesteps_in):
            mask_ts = mask.copy()
            ignores = np.zeros(shape=(self.features_in), dtype=bool)
            for i in self.ignore:
                ignores[i] = True
            mask_ts[i] = ignores
            masks.append(mask_ts)

        for timestep_in,mask in enumerate(masks):
            masked_x = mask*x
            y_masked = self.predict_fn(masked_x)
            mask_out = np.zeros(shape=(self.timesteps_out, len(y_masked[0])), dtype=np.int64)
            mask_out_target = np.zeros(shape=(self.timesteps_out, self.target_features), dtype=np.int64)
            
            for timestep_out in range(0,self.timesteps_out):
                y_masked_fixed_size = mask_out.copy()
                y_masked_fixed_size[timestep_out] = tensor(y_masked[timestep_out]).cpu()
                
                target_fixed_size = mask_out_target.copy()
                target_fixed_size[timestep_out] = tensor(target[timestep_out]).cpu()

                #print(from_numpy(target_fixed_size))
                loss = self.loss(from_numpy(y_masked_fixed_size).float(), from_numpy(target_fixed_size).long()) 
                losses[timestep_in][timestep_out] = loss
        
        plt.figure(figsize = (10,8))
        ys = [str(i) for i in range(-1*self.timesteps_in, 0)]
        xs = [str(i) for i in range(0,self.timesteps_out)]
        ax = sns.heatmap(losses, linewidth=0, xticklabels=xs, yticklabels=ys)
        ax.set_title("HeatMap of importance from input to specific outputs")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Input")
        plt.show()
        return

if __name__=="__main__":
    pass

