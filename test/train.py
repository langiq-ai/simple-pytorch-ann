import torch
import sys
import logging

sys.path.append("../src")

from model import SimpleANN

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.debug("Starting the script")

# Example of how to use the model
input_size = 1000  # Number of input features
hidden_size = 5  # Number of hidden neurons
output_size = 2  # Number of output classes

logging.debug(f"Input size: {input_size}")
logging.debug(f"Hidden size: {hidden_size}")
logging.debug(f"Output size: {output_size}")

# Create the model
model = SimpleANN(input_size, hidden_size, output_size)
logging.debug("Model created successfully")

# Example input
input_data = torch.randn(1, input_size)
logging.debug(f"Input data: {input_data}")

# Forward pass
output = model(input_data)
logging.debug(f"Model output: {output}")

print(f"Model output: {output}")
logging.debug("Script finished successfully")
