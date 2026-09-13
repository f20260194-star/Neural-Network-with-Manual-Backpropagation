Running #1.2 with inputs
net = Network(in_features=64, hidden_features=32, out_features=10)
X_test = numpy.random.randn(5, 64)
output = net.forward(X_test)
print(output.shape)

output (5,10)
