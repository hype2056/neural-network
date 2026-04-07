import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('train.csv', delimiter=',', skiprows=1)

# shuffle so we don't train on ordered data
np.random.shuffle(data)

# split into training and development sets
dev_data = data[:1000].T
train_data = data[1000:].T

# separate labels from pixels, normalize pixels to 0-1
Y_dev = dev_data[0].astype(int)
X_dev = dev_data[1:] / 255.0

Y_train = train_data[0].astype(int)
X_train = train_data[1:] / 255.0

def init_params():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def forward_pass(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_pass(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = np.zeros((10, m))
    one_hot_Y[Y, np.arange(m)] = 1

    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (Z1 > 0)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

def get_accuracy(A2, Y):
    predictions = np.argmax(A2, axis=0)
    return np.mean(predictions == Y)

def train(X, Y, learning_rate, iterations):
    W1, b1, W2, b2 = init_params()
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_pass(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_pass(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if i % 100 == 0:
            print(f"Iteration {i} | Accuracy: {get_accuracy(A2, Y):.2%}")
    
    return W1, b1, W2, b2

W1, b1, W2, b2 = train(X_train, Y_train, 0.1, 500)

def predict(X, Y, index, W1, b1, W2, b2):
    image = X[:, index, None]
    _, _, _, A2 = forward_pass(W1, b1, W2, b2, image)
    prediction = np.argmax(A2)
    label = Y[index]
    
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Prediction: {prediction} | Actual: {label}')
    plt.show()

predict(X_dev, Y_dev, 0, W1, b1, W2, b2)
predict(X_dev, Y_dev, 1, W1, b1, W2, b2)
predict(X_dev, Y_dev, 2, W1, b1, W2, b2)
predict(X_dev, Y_dev, 3, W1, b1, W2, b2)

print(X_train.shape)