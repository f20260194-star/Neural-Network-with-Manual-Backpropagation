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

class ReLU:
    def forward(self, Z):
        self.Z = Z
        return numpy.maximum(0, Z)
