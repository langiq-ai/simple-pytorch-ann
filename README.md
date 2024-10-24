# Simple ANN in PyTorch

This project implements a simple Artificial Neural Network (ANN) using PyTorch. The model is designed with one input layer, one hidden layer, and one output layer. It demonstrates how to perform forward and backward passes, calculate loss, and update weights using an optimizer.

## Requirements

- Python 3.x
- PyTorch
- Torchvision (optional, for dataset utilities)
- Numpy (optional, for data handling)

You can install the dependencies using pip:

```bash
pip install torch torchvision numpy
```

## Model Architecture
### The model architecture is as follows:

* Input Layer: Accepts n input features.
* Hidden Layer: A fully connected layer with ReLU activation.
* Output Layer: A fully connected layer that outputs the predictions. The output can be modified based on the task (e.g., classification or regression).


### Files
* src/model.py: Contains the implementation of the ANN model.
* test/train.py: Demonstrates how to train the model, calculate the loss, and perform backpropagation.
* README.md: This file, which provides an overview of the project.

### Explanation:
- **Requirements**: Lists the required dependencies for the project.
- **Model Architecture**: Provides a high-level description of the model's layers.
- **Usage**: Explains how to run the project, including installation and running steps.
- **Example**: Demonstrates how to initialize and use the model.
- **License**: License information for the project.
- **Contributions**: Encourages contributions.
- **Contact**: Your details for contact. You can modify these as needed.

## Usage
### To run the project, follow these steps:

1. Clone the repository:
```
git clone https://github.com/langiq-ai/simple-pytorch-ann.git
cd simple-ann-pytorch
```

2. Modify the input size, hidden layer size, and output size in the model as needed, depending on your data.

3. Run the training script:
```
python train.py
```
Example code for training can be found in train.py.

## Example 

```
import torch
from model import SimpleANN

# Example of how to use the model
input_size = 10    # Number of input features
hidden_size = 5     # Number of hidden neurons
output_size = 2     # Number of output classes

# Create the model
model = SimpleANN(input_size, hidden_size, output_size)

# Example input
input_data = torch.randn(1, input_size)

# Forward pass
output = model(input_data)

print(f"Model output: {output}")

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request if you want to improve the project or add new features.


### Explanation:
- **Requirements**: Lists the required dependencies for the project.
- **Model Architecture**: Provides a high-level description of the model's layers.
- **Usage**: Explains how to run the project, including installation and running steps.
- **Example**: Demonstrates how to initialize and use the model.
- **License**: License information for the project.
- **Contributions**: Encourages contributions.
- **Contact**: Your details for contact. You can modify these as needed.
