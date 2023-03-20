import numpy as np

class DenseSoftmax:

    def __init__(self, x_len, nodes):
        self.W = np.random.randn(x_len, nodes) 
        self.W /= np.abs(self.W)
        # self.W = np.zeros((x_len, nodes))
        self.dL_dw =  np.zeros((x_len, nodes)) 
        
        print('Number of weight:', np.shape(self.W))

    def forward(self, x):
        self.last_x_shape = x.shape
        self.last_x = x

        z = np.dot(x, self.W)
        z = z - np.max(z)
        self.last_z = z

        e_z = np.exp(np.clip(z, -500, 500))
        softm = e_z / np.sum(e_z, axis=0)
        return softm
        
    def backprop(self, dL_dg, learning_rate, i):

        gradient = dL_dg[i]*np.ones(10)
        # gradient = dL_dg

        e_z = np.exp(np.clip(self.last_z, -500, 500))

        # Sum of all e^z
        S = np.sum(e_z)

        # 1. Gradients of out = g(z) = softmax(z) w.r.t (z)
        # 89%
        dg_dz = -e_z[i] * e_z / (S ** 2) 
        dg_dz[i] = e_z[i] * (S - e_z[i]) / (S ** 2)

        # 2. Gradients of z w.r.t W / x
        dz_dw = self.last_x
        dz_dx = self.W

        # 3. Gradients of loss w.r.t z
        dL_dz = gradient * dg_dz

        # 4. Gradients of loss w.r.t W / x
        self.dL_dw += dz_dw[np.newaxis].T @ dL_dz[np.newaxis] # a @ b: matrix multiplication
        dL_dx = dz_dx @ dL_dz

        # Update W
        # self.W -= learning_rate * dL_dw

        return dL_dx.reshape(self.last_x_shape)
    