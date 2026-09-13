import numpy
import random

#1.1

#Defining architecture

class linear:
    def __init__(self, in_features, out_features):
        self.W = numpy.random.randn(in_features, out_features) * numpy.sqrt(1.0 / in_features)
        self.b = numpy.zeros(out_features)   

    def forward(self, X):
        self.X = X   
        output = X @ self.W + self.b
        return output

class ReLU:                                    #Defining Rectified linear unit
    def forward(self, Z):
        self.Z = Z
        return numpy.maximum(0, Z)

#1.2

class Network:
    def __init__(self, in_features, hidden_features, out_features):
        self.layer1 = linear(in_features, hidden_features)
        self.ReLU = ReLU()
        self.layer2 = linear(hidden_features, out_features)


    def forward(self, X):
        z1 = self.layer1.forward(X)
        a1 = self.ReLU.forward(z1)
        z2 = self.layer2.forward(a1)
        return z2
