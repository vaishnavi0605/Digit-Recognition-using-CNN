from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import correlate2d
import tensorflow.keras as keras
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import tkinter as tk
from PIL import Image, ImageDraw

class Convolution:
    #define the constructor
    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_width = input_shape
        self.input_shape = input_shape
        self.num_filters = num_filters

        #defining the output and filter size
        self.filter_shape = (num_filters, filter_size, filter_size)
        self.output_shape = (num_filters, input_height - filter_size + 1, input_width - filter_size + 1)

        #initialize the filters and biases
        self.filters = np.random.randn(*self.filter_shape)
        self.biases = np.random.randn(*self.output_shape)

    #define the forward pass
    def forward(self, input_data):
        self.input_data = input_data 

        #initialize the output
        output = np.zeros(self.output_shape)
        #Apply the convolution operation
        for i in range(self.num_filters):
            output[i] = correlate2d(input_data, self.filters[i], mode='valid') + self.biases[i]
        #Apply the activation function(ReLU)
        output = np.maximum(0, output)
        return output
    
    #define the backward pass
    def backward(self, dL_dout, lr):
        #initialize the matrices to store gradients
        dL_dinput = np.zeros_like(self.input_data)
        dL_dfilters = np.zeros_like(self.filters)

        #Finding gradients wrt to input and filters
        for i in range(self.num_filters):
            #Calculating the gradient of loss wrt filters
            dL_dfilters[i] = correlate2d(self.input_data, dL_dout[i], mode='valid')
            #Calculating the gradient of loss wrt input
            dL_dinput += correlate2d(dL_dout[i], self.filters[i], mode='full')

        #Update the filters and biases
        self.filters -= lr * dL_dfilters
        self.biases -= lr * dL_dout

        return dL_dinput
    
class MaxPool:
    #define constructor
    def __init__(self, pool_size):
        self.pool_size = pool_size

    #define the forward pass
    def forward(self, input_data):
        self.input_data = input_data
        #define dimensions of pooling layer
        self.num_channels, input_height, input_width = input_data.shape
        self.output_height = input_height // self.pool_size
        self.output_width = input_width // self.pool_size
        #defining the output shape
        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))

        #Iterate over channels
        for c in range(self.num_channels):
            #Loop through height
            for h in range(self.output_height):
                #Loop through width
                for w in range(self.output_width):
                    #Apply max pooling operation
                    self.output[c, h, w] = np.max(input_data[c, h*self.pool_size:(h+1)*self.pool_size, w*self.pool_size:(w+1)*self.pool_size])
        return self.output
    
    #define the backward pass
    def backward(self, dL_dout, lr):
        dL_dinput = np.zeros_like(self.input_data)
        #Iterate over channels
        for c in range(self.num_channels):
            #Loop through height
            for h in range(self.output_height):
                #Loop through width
                for w in range(self.output_width):
                    
                    mask = self.input_data[c, h*self.pool_size:(h+1)*self.pool_size, w*self.pool_size:(w+1)*self.pool_size] == np.max(self.input_data[c, h*self.pool_size:(h+1)*self.pool_size, w*self.pool_size:(w+1)*self.pool_size])
                    dL_dinput[c, h*self.pool_size:(h+1)*self.pool_size, w*self.pool_size:(w+1)*self.pool_size] = mask * dL_dout[c, h, w]
        return dL_dinput
    
class Fully_Connected:
    #define constructor
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        #initialize weights and biases
        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.randn(output_size,1)
    
    #define the activation function (Softmax)
    def softmax(self, z):
        #shift the input values to prevent numerical instability
        z -= np.max(z)
        return np.exp(z) / np.sum(np.exp(z), axis=0)
    #define the derivative of softmax
    #This funtion returns a metrix which store all the derivatives the softmax wrt to all entries
    def softmax_derivative(self, s):
        return np.diagflat(s) - np.dot(s, s.T)
    
    #Define the forward pass
    def forward(self, input_data):
        self.input_data = input_data
        #Flatten the input data
        flattened_input = input_data.flatten().reshape(-1,1)
        #Calculate the output
        self.z = np.dot(self.weights, flattened_input) + self.biases
        #apply softmax
        self.output = self.softmax(self.z)
        return self.output
    
    #define the backward pass
    def backward(self, dl_out, lr):
        #Calculate the gradient of the loss with respect to the pre-activation output
        dL_dy = np.dot(self.softmax_derivative(self.output), dl_out)
        #calculate the gradient of loss wrt weights
        dL_dw = np.dot(dL_dy, self.input_data.flatten().reshape(1,-1))
        #calculate the gradient of loss wrt biases
        dL_db = dL_dy
        #calculate the gradient of loss wrt input
        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.input_data.shape)
        #update the weights and biases
        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db
        return dL_dinput
    
#define the cross entropy loss
def cross_entropy_loss(predictions, target):
    num_samples = 10

    #adding epsilon to avoid numerical instability
    epsilon = 1e-7
    #limit the data into the range of epsilon and 1-epsilon
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    #calculate the loss
    loss = -np.sum(target * np.log(predictions)) / num_samples
    return loss    
#define the derivative of cross entropy loss
def cross_entropy_loss_derivative(actual_labels, predicted_labels):
    num_samples = actual_labels.shape[0]
    gradient = -actual_labels/ (predicted_labels+1e-7) / num_samples
    return gradient

#define a function to train the model
def train_network(X, y, conv, pool, full, lr = 0.01, epochs=200):
    #iterate over the number of epochs
    for epoch in range(epochs):
        #initialize the loss and correct predictions
        total_loss = 0.0
        correct_predictions = 0

        #iterate over the number of samples
        for i in range(len(X)):
            #forward pass
            conv_out = conv.forward(X[i])
            pool_out = pool.forward(conv_out)
            full_out = full.forward(pool_out)
            loss = cross_entropy_loss(full_out.flatten(), y[i])
            total_loss += loss

            #converting the predictions to binary
            one_hot_pred = np.zeros_like(full_out)
            one_hot_pred[np.argmax(full_out)] = 1
            one_hot_pred = one_hot_pred.flatten()

            #getting the index of prediction
            num_pred = np.argmax(one_hot_pred)
            num_y = np.argmax(y[i])
    
            #checking if the prediction is correct
            if num_pred == num_y:   
                correct_predictions += 1

            #Backward pass
            gradient = cross_entropy_loss_derivative(y[i], full_out.flatten()).reshape((-1,1))
            full_back = full.backward(gradient, lr)
            pool_back = pool.backward(full_back, lr)
            conv_back = conv.backward(pool_back, lr)

        #printing the loss and accuracy
        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X) *100.0
        print(f'Epoch: {epoch}, Loss: {average_loss}, Correct Prediction: {correct_predictions}, Accuracy: {accuracy}')

def predict(input_sample, conv, pool, full):
    #forward pass
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)
    #Flattening
    flattened_output = pool_out.flatten()
    #forward pass to fully connected layer
    prediction = full.forward(flattened_output)
    return prediction

#Load the dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

print(train_images[0])
# plot first few images
for i in range(2):
    # define subplot
    plt.subplot(220 + 1 + i)
    # plot raw pixel data
    plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))
    #write the label of the image
    plt.title(train_labels[i])
# show the figure
plt.show()
#Normalize the data
X_train = train_images[:7000] / 255.0
y_train = train_labels[:7000]
X_test = test_images[7000:10000] / 255.0
y_test = test_labels[7000:10000]
#print the shape of X_train
#print(X_train.shape)

#Reshape the data
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#print(y_test[0])
size_filter = int(input('Enter the filter size: '))
num_filters = int(input('Enter the number of filters: '))
pool_size = int(input('Enter the pool size: '))
conv = Convolution(X_train[0].shape, size_filter, num_filters)
pool = MaxPool(pool_size)
full = Fully_Connected((((28-size_filter+1)//pool_size)**2)*num_filters, 10)
#train the network
print('Training the network...')
train_network(X_train, y_train, conv, pool, full)

#predict the output
predictions = []

for data in X_test:
    pred = predict(data, conv, pool, full)
    one_hot_pred = np.zeros_like(pred)
    one_hot_pred[np.argmax(pred)] = 1
    predictions.append(one_hot_pred.flatten())

predictions = np.array(predictions)

#calculate the accuracy
print(accuracy_score(y_test, predictions)*100)

#define a function to take input of image from user
def take_input():
    # Create a new 28x28 image
    image = Image.new('RGB', (28, 28), 'white')
    draw = ImageDraw.Draw(image)

    # Create a new Tkinter window
    window = tk.Tk()

    # Create a canvas for drawing
    canvas = tk.Canvas(window, width=280, height=280, bg='white')
    canvas.pack()

    def draw_image(x, y):
        # Scale the coordinates (since the canvas is 10x the size of the image)
        x //= 10
        y //= 10
        # Draw on the image
        draw.rectangle([x, y, x + 1, y + 1], fill='black')
        # Update the canvas
        canvas.create_rectangle(x * 10, y * 10, x * 10 + 10, y * 10 + 10, fill='black')

    # Bind the drawing function to mouse motion
    canvas.bind('<B1-Motion>', lambda event: draw_image(event.x, event.y))

    def save_image():
        # Save the image
        image.save('drawing.png')

    # Add a button to save the image
    button = tk.Button(window, text='Save', command=save_image)
    button.pack()

    # Run the Tkinter event loop
    window.mainloop()
    return 'drawing.png'

#funtion to take image and make prediction
def make_predictions(image):
    #load the image
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img / 255.0
    #make prediction
    pred = predict(img, conv, pool, full)
    print(f'Predicted label: {np.argmax(pred)}')
    




            
        

        




            
        