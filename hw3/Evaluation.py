
import numpy as np

class Evaluation:
	def Accuracy(self, y_test, y_pred):
		accuracy = np.sum(y_test == y_pred) / len(y_test)
		return accuracy

	def TP(self, y_test, y_pred):
		true_pos = np.sum(np.logical_and(y_pred == 1, y_test == 1))
		return true_pos
	
	def FP(self, y_test, y_pred):
		false_pos = np.sum(np.logical_and(y_pred == 1, y_test == 0)) 
		return false_pos
	
	def TN(self, y_test, y_pred):
		true_neg = np.sum(np.logical_and(y_pred == 0, y_test == 0))
		return true_neg
	
	def FN(self, y_test, y_pred):
		false_neg = np.sum(np.logical_and(y_pred == 0, y_test == 1))
		return false_neg
	
	def Precision(self, y_test, y_pred):
		TP = self.TP(y_test, y_pred)
		FP = self.FP(y_test, y_pred)
		precision = TP / (TP + FP)
		return precision

	def Recall(self, y_test, y_pred):
		TP = self.TP(y_test, y_pred)
		FN = self.FN(y_test, y_pred)
		recall = TP / (TP + FN)
		return recall
