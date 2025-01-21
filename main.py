# Importing the necessary libraries for MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import torchvision
import torchvision.transforms as transforms
import numpy
import matplotlib.pyplot as plt
import seaborn

# Additional libraries which help us create a GUI for the MNIST model
import tkinter
from PIL import Image, ImageGrab

# Checking to see if a GPU is available, but defaults to CPU if not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Defining transformations, what we are doing here is making a preprocessing pipeline to transform the raw MNIST images into a format that is suitable for a deep learning model
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) # ToTensor() converts the image into a tensor, and Normalize() standardizes the image to have a mean of 0.5 and a standard deviation of 0.5

# Loading the MNIST training dataset and the MNIST test dataset
MNIST = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True,)
MNIST_test = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True,)

# Defining dataloading variables, we can use torch.utils.data.DataLoader to load the data in batches instead of at one image at a time (in case we decide to train with a GPU which processes data in parallel)
batch_size = 64 # The number of images that will be loaded in each iteration
train_loader = torch.utils.data.DataLoader(MNIST, batch_size=batch_size, shuffle=True) # Loading MNIST training data
test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=batch_size, shuffle=False) # Loading MNIST test data

# When using new unfamiliar datasets, it is always a good idea to visualize the data to get a better understanding of it, we can do this using seaborn
# Visualizing the distribution of labels in our training set to ensure they are evenly distributed
seaborn.countplot(x=numpy.array(MNIST.targets))
plt.title('Distribution of Labels in Training Set')
plt.show()

# When using new unfamiliar datasets, it is always a good idea to check for missing values in the dataset, or NaN values, we can do this using numpy
# Here we are looking to find any NaN values in our data, ideally both print statements should return False
print(torch.isnan(MNIST.data.float()).any()) # Checking for NaN values in the training data
print(torch.isnan(MNIST_test.data.float()).any()) # Checking for NaN values in the

# Defining a Convolutional Neural Network (CNN) model
class MNISTModel(nn.Module):

    # Initializing the model, and defining all the layers of our CNN's Sequential Model
    def __init__(self): # The __init__ function allows us to define our architecture, and create layers BEFORE we train the model
        super(MNISTModel, self).__init__()

        # Defining first 2 convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2) # Our first convolutional layer, we are using 32 5x5 pixel filters to scan over the image and detect patterns. At this layer we define our input shape as well
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2) # This is the same as the first one, but the 32 represents input channels, as it is recieving 32 input channels from the previous layer

        # Defining max pooling layer (reducing image size by a factor of 2)
        self.pool = nn.MaxPool2d(2, 2) # The Max Pooling layer reduces the image size by taking the maximum value of each 2x2 block, reducing our image from 28x28 to 14x14

        # Defining dropout layer to prevent overfitting
        self.dropout1 = nn.Dropout(0.25) # This temporarily drops 25% of neurons in the layer, to prevent overfitting

        # Defining second 2 convolutional layers
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Since our image is now 14x14, we can use smaller filters to detect finer details and patterns (kernel represents grid size)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # padding='same' keeps the output the same size as the input


        # Defining another dropout layers
        self.dropout2 = nn.Dropout(0.25) # Drops 25% of neurons

        # Defining fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # A fully connected layer with 128 neurons, each in charge of learning a complex combination of patterns and previous features
        self.dropout3 = nn.Dropout(0.4) # Another dropout layer, this time dropping 50% of neurons
        self.fc2 = nn.Linear(128, 10) # Our final layer, with 10 neurons, one for each number 0-9

    # Defining the forward pass of our model, this is where we define how the data flows through our model
    # While Tensorflow assumes a sequential model, PyTorch does not, so we need to use this to maintain order, and customize the functions and outputs of each individual layer
    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = functional.relu(self.conv3(x))
        x = functional.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1) # Flattening the image into a 1D tensor automatically
      # x = x.view(-1, 64 * 7 * 7) # This is the same as the line above, but more explicit, and relies on the image size beign 7 x 7
        x = functional.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)

        return x
    
# Instantiating the model
model = MNISTModel().to(device)

# Defining the loss function and optimizer
criterion = nn.CrossEntropyLoss() # This is our loss function, effectively measuring how badly our model performed
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08, momentum=0.9) # The optimizer is in charge of updating the model's weights based on it's loss. RMSprop is a good optimizer that adapts the learning rate for each parameter

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.to(device)
    model.train()
    train_loss, train_acc = [], [] # 2 arrays to track the loss values and the accuracy of our model
    
    for epoch in range(epochs):
        runningLoss = 0.0
        correct = 0
        total = 0

        for images, labels, in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = runningLoss / len(train_loader)
        epoch_acc = correct / total

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        if epoch_acc >= 0.995: # If the model reaches 99.5% accuracy, we stop training
            print("Model has reached 99.5% accuracy, stopping training")
            break

    return train_loss, train_acc

# Defining number of epochs, and starting to train our model
epochs = 5
train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, epochs)

# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    runningLoss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            runningLoss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    test_loss = runningLoss / len(test_loader)
    test_acc = correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

test_accuracy = evaluate_model(model, test_loader)

# The following code will allow us to visualize the loss and accuracy of our model through Loss & Accuracy Curves
# -- IT IS COMMENTED OUT FOR SIMPLICITY SAKE, BUT GOOD TO USE WHEN TRAINING MODELS --
# fig, ax = plt.subplots(2,1)
# ax[0].plot(train_loss, color='b', label="Training Loss")
# ax[0].legend(loc='best', shadow=True)
# ax[0].set_title("Training Loss Curve")

# ax[1].plot(train_acc, color='r', label="Training Accuracy")
# ax[1].legend(loc='best', shadow=True)
# ax[1].set_title("Training Accuracy Curve")

# plt.tight_layout()
# plt.show()

torch.save(model.state_dict(), 'mnist_model.pt') # Saving the model's weights to a file



# ------------------- EVERYTHING BELOW THIS LINE IS NOT IN THE SCOPE OF THE MNIST WORKSHOP, IT IS A SIMPLE CODE USED TO CREATE A GUI FOR THE MNIST MODEL -------------------
# Code is taken from Aly Ashour (President of WEAP), and slightly modified to fit the display settings of both Mac & Windows

def run_gui():
    # Load the PyTorch model
    model = MNISTModel()
    model.load_state_dict(torch.load('mnist_model.pt', weights_only=True))
    model.eval()

    # test the model before we run the GUI
    device = torch.device("cpu")
    test_dataset = torchvision.datasets.MNIST('../data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1000)
    evaluate_model(model, test_loader)

    # Run the GUI
    root = tkinter.Tk()
    root.title("MNIST DIGIT CLASSIFIER")

    canvas = tkinter.Canvas(root, width=300, height=300, bg='white')
    canvas.pack()

    def preprocess(img):
        # resize, grayscale, and flip black and white
        img = img.resize((28, 28)).convert('L')
        img_array = numpy.array(img)
        img_array = 255 - img_array
        img = Image.fromarray(img_array.astype(numpy.uint8))

        # apply the same transformation as before
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0) # add a dimension for the batch

        # debug
        # img.show()

        return img_tensor

    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')

    is_retina = hasattr(tkinter, "scaling") and tkinter.scaling() > 1.0

    def predict_digit():
        x0 = (root.winfo_rootx() + canvas.winfo_x())
        y0 = (root.winfo_rooty() + canvas.winfo_y())
        x1 = x0 + canvas.winfo_width()
        y1 = y0 + canvas.winfo_height()

        if is_retina:
            x0, y0, x1, y1 = 2 * x0, 2 * y0, 2 * x1, 2 * y1

        img = ImageGrab.grab().crop((x0, y0, x1, y1))
        
        # preprocess and predict
        img_tensor = preprocess(img).to(device)
        with torch.no_grad():
            prediction = model(img_tensor)
            probabilities = functional.softmax(prediction, dim=1)
            digit = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()

        # display result
        result_label.config(text=f'Prediction: {digit}\nConfidence: {confidence:.2f}')

    def clear_canvas():
        canvas.delete('all')

    canvas.bind("<B1-Motion>", paint)

    btn_predict = tkinter.Button(root, text="Predict", command=predict_digit)
    btn_predict.pack()

    btn_clear = tkinter.Button(root, text="Clear", command=clear_canvas)
    btn_clear.pack()

    result_label = tkinter.Label(root, text="", font=("Helvetica", 16))
    result_label.pack()

    root.mainloop()

if __name__ == '__main__':
    run_gui()