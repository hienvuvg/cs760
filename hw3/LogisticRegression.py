# %%
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LogisticRegression:
	def __init__(self, n_features):
		self.weight = np.zeros((n_features,1))
		self.bias = 0

	def normalization(self, X):		
		scaler = MinMaxScaler()
		X = scaler.fit_transform(X)
		return X

	# The Sigmoid or Logistic function: squishes all its inputs between 0 and 1
	# Input: all Real Numbers
	# Output: 0 to 1
	def sigmoid(self, z):
		return 1.0/(1 + np.exp(-z))

	# Calculate Cross-Entropy Loss
	def loss(self, y, y_hat):
		return -np.mean(y*(np.log(y_hat)) + (1-y)*np.log(1-y_hat))

	# The gradient of the cross-entropy loss for logistic regression with respect to weight and bias
	# Ref 1: https://web.stanford.edu/~jurafsky/slp3/5.pdf
	# Ref 2: https://medium.com/analytics-vidhya/logistic-regression-with-gradient-descent-explained-machine-learning-a9a12b38d710
	def gradient(self, X_train, y_train, y_hat):
		n = len(y_train) # No of traning samples

		d_theta_1 = (1/n)*np.dot(X_train.T, (y_hat - y_train))
		d_theta_0 = (1/n)*np.sum((y_hat - y_train)) 
		
		return d_theta_1, d_theta_0

	def fit(self, X_train, y_train, epochs, lr):

		X_train = self.normalization(X_train)

		y_train = y_train.reshape(len(y_train),1)

		loss = []
		for i in range(epochs):
			# Calculate the hypothesis
			y_hat = self.sigmoid(np.dot(X_train, self.weight) + self.bias)
			
			# Getting the gradients of loss w.r.t parameters.
			d_theta_1, d_theta_0 = self.gradient(X_train, y_train, y_hat)
			
			self.weight = self.weight - lr*d_theta_1
			self.bias 	= self.bias - lr*d_theta_0
			
			loss.append(self.loss(y_train, y_hat))

		return loss

	def predict_proba(self, X_test):
		X_test = self.normalization(X_test)
		y_pred_proba = self.sigmoid(np.dot(X_test, self.weight) + self.bias) # Calculate the hypothesis
		
		return np.concatenate((y_pred_proba, y_pred_proba), axis = 1)
	
	def predict(self, X_test):
		X_test = self.normalization(X_test)
		predictions = self.sigmoid(np.dot(X_test, self.weight) + self.bias) # Calculate the hypothesis
		y_pred = np.asarray([1 if y_hat >= 0.5 else 0 for y_hat in predictions])
		
		return y_pred
    
# %%
