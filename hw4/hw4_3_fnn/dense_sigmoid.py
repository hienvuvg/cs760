import numpy as np

class DenseSigmoid:

    def __init__(self, x_len, nodes):
        self.W = np.random.randn(x_len, nodes)
        self.W /= np.abs(self.W)
        # self.nabla_w = np.zeros((x_len, nodes))
        self.dL_dw =  np.zeros((x_len, nodes)) 

        print('Number of weight:', np.shape(self.W))

    def forward(self, x):

        self.last_x_shape = x.shape
        self.last_x = x

        z = np.dot(x, self.W)
        self.last_z = z

        e_nz = np.exp(np.clip(-z, -500, 500))
        sigmoid = 1 / (1 + e_nz)
        return sigmoid

    def backprop(self, dL_dg, learning_rate):

        gradient = dL_dg

        # z = z = ax + b
        e_nz = np.exp(np.clip(-self.last_z, -500, 500))
        sigmoid = 1/(1+e_nz)

        # 1. Gradients of out = sigmoidoid(z) w.r.t z
        dg_dz = sigmoid*(1 - sigmoid)

        # 2. Gradients of z w.r.t W / x
        # z = w * x + b
        dz_dw = self.last_x
        dz_dx = self.W

        # 3. Gradients of loss w.r.t z
        dL_dz = gradient * dg_dz

        # 4. Gradients of loss w.r.t W / x
        self.dL_dw += dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
        dL_dx = dz_dx @ dL_dz 

        # Update W / b
        # self.W -= learning_rate * dL_dw

        return dL_dx.reshape(self.last_x_shape)
