import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchsummary import summary

from model_1 import CNN_Model_1, get_optimizer as get_optimizer_1, get_optimizer_params as get_optimizer_params_1
from model_2 import CNN_Model_2, get_optimizer as get_optimizer_2, get_optimizer_params as get_optimizer_params_2
try:
    from model_3 import CNN_Model_3, get_optimizer as get_optimizer_3, get_optimizer_params as get_optimizer_params_3
    MODEL_3_AVAILABLE = True
except ImportError:
    MODEL_3_AVAILABLE = False
    print("Note: model_3.py not found. Only model_1 and model_2 are available.")

from utils import compute_mean_std, data_loader, initialize_model, save_plot_metrics, save_train_test_metrics, train_model

print("Starting training script...")

# Model selection
print("\n" + "="*50)
print("Model Selection")
print("="*50)
print("Available models:")
print("1. CNN_Model_1 - Basic CNN Architecture")
print("2. CNN_Model_2 - CNN with BatchNorm and Dropout")
if MODEL_3_AVAILABLE:
    print("3. CNN_Model_3 - Optimized CNN Architecture")

while True:
    try:
        if MODEL_3_AVAILABLE:
            choice = input("\nSelect model to train (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                break
        else:
            choice = input("\nSelect model to train (1/2): ").strip()
            if choice in ['1', '2']:
                break
        print("Invalid choice. Please select a valid option.")
    except KeyboardInterrupt:
        print("\nTraining cancelled by user.")
        exit(0)

# Map choice to model class, name, and optimizer functions
model_mapping = {
    '1': (CNN_Model_1, 'CNN_Model_1', get_optimizer_1, get_optimizer_params_1),
    '2': (CNN_Model_2, 'CNN_Model_2', get_optimizer_2, get_optimizer_params_2),
}

if MODEL_3_AVAILABLE:
    model_mapping['3'] = (CNN_Model_3, 'CNN_Model_3', get_optimizer_3, get_optimizer_params_3)

selected_model_class, selected_model_name, get_optimizer_func, get_optimizer_params_func = model_mapping[choice]
print(f"\nSelected: {selected_model_name}")
print("="*50)

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
optimizer_func = get_optimizer_func()
optimizer_params = get_optimizer_params_func()
model, criterion, optimizer, device = initialize_model(selected_model_class, optimizer_func, **optimizer_params)
print("Model initialized successfully.")

# Print model summary
print("Model Summary:")
summary(model, input_size=(1, 28, 28)) # Since input size for MNIST is 28x28 with 1 channel

# Model Training
print(f"Starting training for {selected_model_name}...")
train_losses, train_accuracies, test_losses, test_accuracies = train_model(model, device, train_loader, test_loader, optimizer, 
                                                                           criterion, epochs=20)
print("Training completed!")

# Save the train and test losses and accuracies for future reference
print("Saving training and testing metrics...")
save_train_test_metrics(train_losses, train_accuracies, test_losses, test_accuracies, selected_model_name)
print("Metrics saved successfully.")

# Save plot metrics for visualization
print("Saving plot metrics...")
save_plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies, selected_model_name)
print("Plot metrics saved successfully.")

# Print final statistics
print(f"\nFinal Results:")
print(f"Training Loss: {train_losses[-1]:.4f}")
print(f"Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Test Loss: {test_losses[-1]:.4f}")
print(f"Test Accuracy: {test_accuracies[-1]:.2f}%")
