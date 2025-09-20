import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchsummary import summary

from model_1 import CNN_Model_1
from utils import compute_mean_std, data_loader, initialize_model, save_plot_metrics, save_train_test_metrics, train_model

print("Starting training script...")

# Check if CUDA is available
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# Compute mean and std of training data
print("Computing mean and std of training data...")
train_mean, train_std = compute_mean_std()
print("Computation done.")
print(f"Mean: {train_mean}, Std: {train_std}")

# Data loaders
print("Preparing data loaders...")
train_loader, test_loader = data_loader(train_mean=train_mean, train_std=train_std, batch_size_train=256, batch_size_test=2048)
print("Data loaders ready.")

# Initialize model, loss function, optimizer
print("Initializing model...")
model, criterion, optimizer, device = initialize_model(CNN_Model_1, optimizer='Adam', lr=0.001)
print("Model initialized successfully.")

# Print model summary
print("Model Summary:")
summary(model, input_size=(1, 28, 28)) # Since input size for MNIST is 28x28 with 1 channel

# Model Training
print("Starting training for CNN_Model_1...")
train_losses, train_accuracies, test_losses, test_accuracies = train_model(model, device, train_loader, test_loader, optimizer, 
                                                                           criterion, epochs=20)
print("Training completed!")

# Save the train and test losses and accuracies for future reference
print("Saving training and testing metrics...")
save_train_test_metrics = {
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
    'model_name': 'CNN_Model_1'
}
print("Metrics saved successfully.")

# Save plot metrics for visualization
print("Saving plot metrics...")
save_plot_metrics = {
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
    'model_name': 'CNN_Model_1'
}
print("Plot metrics saved successfully.")

# Print final statistics
print(f"\nFinal Results:")
print(f"Training Loss: {train_losses[-1]:.4f}")
print(f"Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Test Loss: {test_losses[-1]:.4f}")
print(f"Test Accuracy: {test_accuracies[-1]:.2f}%")
