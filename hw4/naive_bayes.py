# %%
import numpy as np

class MultinomialNaiveBayes:
    def __init__(self, X, y, alpha=0.5):
        labels = np.unique(y)
        self.alpha = alpha # the smoothing value
        _, self.n_features = np.shape(X)
        self.n_classes = len(labels)
        self.K_L = self.n_classes
        self.class_values = np.unique(y)
        self.prior_proba = np.zeros(len(labels))
        self.conditional_proba = np.zeros((self.n_features, self.n_classes))
    
    def prior_prob(self, y, c_k): 
        N = len(y) # Number of data samples
        sum_y = 0
        for y_i in y:
            if y_i == c_k:
                sum_y += 1
        p_a = (sum_y + self.alpha) / (N + self.K_L * self.alpha)
        return p_a
    
    def train(self, X_train, y_train):

        y_train = y_train[np.newaxis].T
        train_set = np.concatenate((X_train, y_train), axis = 1)

        # Calculate the prior probability, i.e. P(Y = c_k) where c_k is the class value
        for class_index in range (self.n_classes):
            self.prior_proba[class_index] = self.prior_prob(y_train, self.class_values[class_index])
            # print('P_y[%d] = %.5f'%(class_index, self.prior_proba[class_index]))

        # Calculate the conditional probability, i.e. theta = p_hat(c_i | y=e) where i is the i-th character
        for class_index, y_class in enumerate(np.unique(y_train)):
            feature_value = np.zeros(self.n_features)
            total_value = 0
            for row in train_set:   # Process each data sample separately
                if row[-1] == y_class:  # Only consider sample with the same class value
                    for i, feature in enumerate(row[:-1]):  # Go through each feature of the sample to count the total value
                        feature_value[i] += feature  # Count the total value of that particular feature
                        total_value += feature      # Count the total value of all feature in data samples of the same class

            # Calculate the probability     
            for feature_index in range(self.n_features):
                self.conditional_proba[feature_index, class_index] = (feature_value[feature_index] + self.alpha) / (total_value + self.K_L * self.alpha)
        
        # # Check the calculation
        # print(np.round(self.conditional_proba,3))
        # for j in range(self.n_classes):
        #     prob = 0
        #     for i in range(self.n_features):
        #         prob += self.conditional_proba[i, j]
        #     print(prob) # Must be 1


    def likelihood(self, row, y_class):
        log_p_pred = 0 
        for i, feature in enumerate(row):   # Go through each feature in one sample
            for k in range(feature):        # Accumulate the log probability by the value of the feature
                log_p_pred += np.log(self.conditional_proba[i, y_class])  # Use log to avoid the disminishing problem of multiplying with value < 1
        return log_p_pred
    
    # Prediction for multiple input samples
    def predict(self, X_test):
        y_pred = []
        for row in X_test:
            y_hat = np.zeros(self.n_classes)
            for i, y_class in enumerate(self.class_values): # Go through each class
                y_hat[i] = self.likelihood(row, y_class) + np.log(self.prior_proba[y_class])  # Calculate the log probability for each class, i.e. posterior p_hat(y_i|x) 
            y_pred.append(np.argmax(y_hat)) # Adding the only class with maximum likelihood to the list
            # print(y_hat, end=" ")
            # print(np.argmax(y_hat))
        return np.asarray(y_pred)
    
    # Prediction for a single input sample
    def single_predict(self, X_test):
        y_hat = np.zeros(self.n_classes)
        for i, y_class in enumerate(self.class_values): # Go through each class
            y_hat[i] = self.likelihood(X_test, y_class) + np.log(self.prior_proba[y_class]) # Calculate the log probability for each class, i.e. posterior p_hat(y_i|x) 
        return np.argmax(y_hat)
    
    # Printout posterior probability of a single input sample
    def single_posterior_proba(self, X_test):
        y_hat = np.zeros(self.n_classes)
        for i, y_class in enumerate(self.class_values): # Go through each class
            y_hat[i] = self.likelihood(X_test, y_class) + np.log(self.prior_proba[y_class]) # Calculate the log probability for each class, i.e. posterior p_hat(y_i|x) 
            print('y['+str(i)+']= e^('+str(round(y_hat[i]))+')')
            # p_pred = np.exp(y_hat[i]) # Convert from log to probability
            # print(p_pred)


    