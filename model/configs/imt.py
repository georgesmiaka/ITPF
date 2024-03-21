from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
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
    '''
    def __init__(self, predict_fn, loss, example_x, example_y, features):
        self.predict_fn = predict_fn
        self.loss = loss
        self.example_x = example_x
        self.example_y = example_y
        self.features = features

        # Data related sizes,
        self.timesteps_in = example_x.shape[0] 
        self.timesteps_out = example_y.shape[0]
        self.features_in = example_x.shape[1]
        self.target_features = example_y.shape[1]
        
    
    def avg_feature_importance(self, x):
        '''
        The average feature importance, computed by masking all the features (one-by-one) and comparing the masked prediction to the ground truth.

        x: input to the model on the form (timesteps_in, features_in)
        '''
        target = self.predict_fn(x)
        losses = np.zeros(shape=(self.features_in))
        masks = []
        mask = np.ones(shape=(self.features_in), dtype=bool)

        for i in range(0,self.features_in):
            mask_feature = mask.copy()
            mask_feature[i] = False
            masks.append(mask_feature)

        for i, mask in enumerate(masks):
            masked_x = np.array(list(map(lambda x_row: x_row * mask, x)))
            y_masked = self.predict_fn(masked_x)
            loss = self.loss(target, y_masked)
            losses[i] = loss
        
        fig, ax = plt.subplots()
        order = np.argsort(losses)
        ax.barh(np.arange(len(losses)), (losses[order] / sum(losses)) * 100, tick_label=np.asarray(self.features)[order])
        ax.set_title("Average Feature Importance")
        ax.set_xlabel("Importance in %")
        plt.plot()

    def timestep_importance(self, x):
        '''
        The timestep importance, computed by masking all the timesteps (one-by-one) and comparing the prediction to the ground truth.

        x: input to the model on the form (timesteps_in, features_in)
        '''
        target = self.predict_fn(x)
        losses = np.zeros(shape=(self.timesteps_in))
        masks = []
        mask = np.ones(shape=(self.timesteps_in, self.features_in), dtype=bool)

        for i in range(0,self.timesteps_in):
            mask_ts = mask.copy()
            mask_ts[i] = np.zeros(shape=(self.features_in), dtype=bool)
            masks.append(mask_ts)

        for i, mask in enumerate(masks):
            masked_x = mask*x
            y_masked = self.predict_fn(masked_x)
            loss = self.loss(target, y_masked)
            losses[i] = loss

        fig, ax = plt.subplots()
        ax.plot(range(-1 * self.timesteps_in, 0), losses)
        ax.set_title("Timestep importance")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Importance")
        plt.plot()

        return
    
    def windowed_feature_importance(self, x, window_size: 10, stride: 10):
        '''
        Averaged feature importance across the sliding window

        x: input to the model on the form (timesteps_in, features_in)
        window_size: size of the sliding window
        stride: how much the window moves every itteration
        '''
        target = self.predict_fn(x)
        mask = np.ones(shape=(self.timesteps_in, self.features_in), dtype=bool)
        window_starts = [i for i in range(0,self.timesteps_in,stride) if i+window_size <= self.timesteps_in]
        losses = np.zeros(shape=(len(window_starts), self.features_in))

        for i,start in enumerate(window_starts):
            masks = []
            for j in range(0,self.features_in):
                masked_window_feature = mask.copy()
                masked_window_feature[start:(window_size+start),j] = np.zeros(shape=(window_size), dtype=bool)
                masks.append(masked_window_feature)
            for feature,wf_mask in enumerate(masks):
                masked_x = wf_mask*x
                y_masked = self.predict_fn(masked_x)
                loss = self.loss(target, y_masked)
                losses[i][feature] = loss

        windows = [str((i,i+window_size)) for i in window_starts]
        plot_ready = pd.DataFrame(losses, columns=self.features)
        plot_ready = plot_ready.astype(float)
        plot_ready.loc[:, "window"] = windows
        plot_ready.plot(x='window', kind='bar', stacked=True, title='Windowed Feature Importance')
        plt.show()
        return
    
    def feature_importance(self, x):
        '''
        The importance of every feature at every timestep, creates a lineplot for all the features.
        Very computational expensive, think masking every single feature at every single timestep.

        x: input to the model on the form (timesteps_in, features_in)
        '''

        target = self.predict_fn(x)
        mask = np.ones(shape=(self.timesteps_in, self.features_in), dtype=bool)
        losses = np.zeros(shape=(self.timesteps_in, self.features_in))

        for i in range(0,self.timesteps_in):
            for j in range(0,self.features_in):
                masked_feature_timestep = mask.copy()
                masked_feature_timestep[i][j] = False
                masked_x = masked_feature_timestep * x
                y_masked = self.predict_fn(masked_x)
                loss = self.loss(target, y_masked)
                losses[i][j] = loss
        
        fig, ax = plt.subplots()
        for feature in (losses.transpose()):
            ax.plot(range(-1 * self.timesteps_in, 0), feature)
        ax.set_title("Feature Importance Across All Timesteps")

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Importance")
        ax.legend(self.features)
        plt.plot()

        return
    
    def output_input_importance(self, x):
        '''
        How masking out inputs relate to specific timesteps in the future, maybe instead we should do some kind of grouping or selecting the 5-10 most important timesteps for every output timestep

        x: input to the model on the form (timesteps_in, features_in)
        '''
        target = self.predict_fn(x)
        losses = np.zeros(shape=(self.timesteps_in, self.timesteps_out))
        masks = []
        mask = np.ones(shape=(self.timesteps_in, self.features_in), dtype=bool)
        for i in range(0,self.timesteps_in):
            mask_ts = mask.copy()
            mask_ts[i] = np.zeros(shape=(self.features_in), dtype=bool)
            masks.append(mask_ts)

        for timestep_in,mask in enumerate(masks):
            masked_x = mask*x
            y_masked = self.predict_fn(masked_x)
            for timestep_out in range(0,self.timesteps_out):
                print(timestep_in, timestep_out, target.shape, y_masked.shape)
                loss = self.loss(target[timestep_out], y_masked[timestep_out])
                losses[timestep_in][timestep_out] = loss
        
        print(losses.shape)
        plt.figure(figsize = (10,8))
        ys = [str(i) for i in range(-1*self.timesteps_in, 0)]
        xs = [str(i) for i in range(0,self.timesteps_out)]
        ax = sns.heatmap(losses, linewidth=0, xticklabels=xs, yticklabels=ys)
        ax.set_title("HeatMap of attention from input to specific outputs")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Input")
        plt.show()

        return

if __name__=="__main__":
    pass

