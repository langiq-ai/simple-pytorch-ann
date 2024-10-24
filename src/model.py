import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.debug("Starting the script")


# Define the simple ANN model
class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleANN, self).__init__()
        logging.debug(
            f"Initializing SimpleANN with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}"
        )
        # Input to hidden layer
        self.hidden = nn.Linear(input_size, hidden_size)
        # Hidden to output layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        logging.debug(f"Forward pass with input: {x}")
        # Apply the hidden layer with ReLU activation
        x = F.relu(self.hidden(x))
        logging.debug(f"After hidden layer: {x}")
        # Output layer with no activation (or softmax for classification tasks)
        x = self.output(x)
        logging.debug(f"After output layer: {x}")
        return x


# Example usage
input_size = 10  # Number of input features
hidden_size = 50  # Number of hidden neurons
output_size = 2  # Number of output classes

logging.debug(
    f"Creating model with input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}"
)
# Create the model
model = SimpleANN(input_size, hidden_size, output_size)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example input (batch size of 1, with 10 features)
input_data = torch.randn(1, input_size)
logging.debug(f"Input data: {input_data}")

# Forward pass
output = model(input_data)
logging.debug(f"Model output: {output}")

# Example target (for loss calculation)
target = torch.tensor([1])  # Example target class
logging.debug(f"Target: {target}")

# Calculate loss
loss = criterion(output, target)
logging.debug(f"Loss: {loss.item()}")

# Backward pass and optimization
loss.backward()
logging.debug("Performed backward pass")
optimizer.step()
logging.debug("Optimizer step completed")

print(f"Output: {output}")
print(f"Loss: {loss.item()}")
logging.debug("Script finished successfully")
