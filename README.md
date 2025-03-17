# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

![Screenshot 2025-03-02 192402](https://github.com/user-attachments/assets/6df48afd-5661-49c4-921b-66a9aa657e86)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SUDHARSANA KUMAR S R
### Register Number: 212223240162
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here
        self.fc1 = nn.Linear(1,5)
        self.fc2 = nn.Linear(5,8)
        self.fc3 = nn.Linear(8,5)
        self.fc4 = nn.Linear(5,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.fc4(x)

    return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
## Dataset Information

![Screenshot 2025-03-17 123943](https://github.com/user-attachments/assets/a7fb09f1-91c3-4f16-b899-d477bd9ecc5f)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-03-17 124054](https://github.com/user-attachments/assets/c36f819e-1b55-4390-9dbe-9b69f53a6d26)





### New Sample Data Prediction

![Screenshot 2025-03-17 124329](https://github.com/user-attachments/assets/2b74acbf-5e98-4521-a703-dd5096324f83)


## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
