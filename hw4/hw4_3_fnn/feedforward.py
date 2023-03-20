import numpy as np
from dense_softmax import DenseSoftmax
from dense_sigmoid import DenseSigmoid

class NN:
    def __init__(self, W1_s, W2_s, W3_s, Y_s):
        self.layer_1 = DenseSigmoid(W1_s, W2_s)
        self.layer_2 = DenseSigmoid(W2_s, W3_s)
        self.layer_3 = DenseSoftmax(W3_s, Y_s)
    
    def predict(self, image, label):

        in_data = image / 255 - 0.5 # Acc of only 33% without normalization, 
        out = self.layer_1.forward(in_data.flatten())
        out = self.layer_2.forward(out)
        p_out = self.layer_3.forward(out)

        # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
        loss = -np.log(p_out[label])
        y_pred = 1 if np.argmax(p_out) == label else 0

        return p_out, loss, y_pred


    def train(self, images, labels, learning_rate=0.001, batch_size=1):

        np.set_printoptions(suppress=True)
        
        losses = 0
        accuracy = 0
        gradient = np.zeros(10)

        # Forward
        index = 0
        for image, label in zip(images, labels):
            p_out, loss, y_pred = self.predict(image, label)
            losses += loss
            accuracy += y_pred

            gradient = np.zeros(10)
            gradient[label] = -1 / p_out[label] 

            # Backpropagation
            gradient1 = self.layer_3.backprop(gradient, learning_rate, label)
            gradient2 = self.layer_2.backprop(gradient1, learning_rate)
            gradient3 = self.layer_1.backprop(gradient2, learning_rate)

            index += 1

            if index == batch_size:
                # self.layer_3.dL_dw /= batch_size
                # self.layer_2.dL_dw /= batch_size
                # self.layer_1.dL_dw /= batch_size

                self.layer_3.W -= learning_rate * self.layer_3.dL_dw
                self.layer_2.W -= learning_rate * self.layer_2.dL_dw
                self.layer_1.W -= learning_rate * self.layer_1.dL_dw

                self.layer_3.dL_dw = np.zeros((200, 10)) 
                self.layer_2.dL_dw = np.zeros((300, 200)) 
                self.layer_1.dL_dw = np.zeros((784, 300)) 

                index = 0

        return losses/len(labels), accuracy/len(labels)
    