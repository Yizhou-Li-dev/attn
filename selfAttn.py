import numpy as np

class SelfAttention():
    def __init__(self, q, k, v, sqr_dim_k=8):
        self.q = q  
        self.k = k
        self.v = v
        # the square root of the dimension of the key vectors used in the paper – 64.
        # This leads to having more stable gradients. 
        # There could be other possible values here, but this is the default
        self.sqr_dim_k = sqr_dim_k

    def forward(self):
        s = self.q @ np.transpose(self.k)
        s = np.divide(s, self.sqr_dim_k)
        s = self.softmax(s) # use np version
        s = s @ self.v # normally there will be other kv adding together... q1,k2,v2 + q1,k3,v3
        return s

    def transpose(self, m):
        _out = len(m)
        _in = len(m[0])
        new_m = [[0 for _ in range(_out)] for _ in range(_in)]
        for i in range(_out):
            for j in range(_in):
                new_m[j][i] = m[i][j]
        return new_m
    
    def softmax(self, m):
        m -= np.max(m)
        m = np.exp(m) # to prevent index overflow
        return m / m.sum()

selfAttention = SelfAttention(q=np.array([[1, 2], [3, 4]]), k=np.array([[1, 2], [3, 4]]), v=np.array([[1, 2], [3, 4]]))
print(selfAttention.forward())

