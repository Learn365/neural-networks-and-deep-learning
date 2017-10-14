"""
regularization
"""
import mnist_loader
import network2

# load test data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784,30,10],cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data[:1000],400,10,0.5,evaluation_data=test_data,
lmbda = 0.1,
monitor_evaluation_cost=True,monitor_evaluation_accuracy=True,
monitor_training_accuracy=True,monitor_training_cost=True)