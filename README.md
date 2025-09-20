# TSAI Assignment 6 - CNN Model Training Experiments

This repository contains three progressive experiments to build and optimize CNN models for MNIST digit classification, following a systematic approach to achieve high accuracy with minimal parameters.

## Project Overview

The goal of this assignment is to train 3 models with the following progressive targets:

### Experiment 1 – Basic Model Setup ✅
**Targets:**
- Build a working CNN skeleton using convolution, pooling, and fully connected layers
- Ensure the model trains without errors
- Keep parameter count under ~100k
- Achieve at least 98.4% test accuracy as a starting baseline
- Train within ≤ 20 epochs

### Experiment 2 – Introduce Regularization & Optimizations
**Targets:**
- Make the model lighter (reduce parameters closer to ~20k)
- Add Batch Normalization, Dropout, and Image Augmentation to improve generalization
- Experiment with proper MaxPooling placement and Global Average Pooling to reduce parameters
- Target: ≥ 99.4% test accuracy within 20 epochs, parameters < 20k

### Experiment 3 – Final Optimized Model
**Targets:**
- Achieve the required benchmark: ≥ 99.4% test accuracy consistently in the last few epochs
- Train within ≤ 15 epochs
- Keep total parameters ≤ 8,000
- Use best practices: proper skeleton, BN, Dropout, GAP, data augmentation, lighter architecture, and tuned learning rate schedule
- Ensure reproducibility of results

---

## Experiment 1 Results - CNN_Model_1 ✅

### Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
            Conv2d-2           [-1, 16, 24, 24]           1,152
            Conv2d-3           [-1, 32, 22, 22]           4,608
         MaxPool2d-4           [-1, 32, 11, 11]               0
            Conv2d-5             [-1, 64, 9, 9]          18,432
            Conv2d-6             [-1, 64, 7, 7]          36,864
            Conv2d-7             [-1, 32, 5, 5]          18,432
            Conv2d-8             [-1, 32, 5, 5]           1,024
            Conv2d-9             [-1, 10, 1, 1]           8,000
================================================================
Total params: 88,584
Trainable params: 88,584
Non-trainable params: 0
----------------------------------------------------------------
```

### Training Logs
```
Epoch [1/20] - Train Loss: 0.3047, Train Acc: 90.16% - Test Loss: 0.0841, Test Acc: 97.28%
Epoch [2/20] - Train Loss: 0.0729, Train Acc: 97.79% - Test Loss: 0.0550, Test Acc: 98.28%
Epoch [3/20] - Train Loss: 0.0553, Train Acc: 98.29% - Test Loss: 0.0487, Test Acc: 98.55%
Epoch [4/20] - Train Loss: 0.0416, Train Acc: 98.71% - Test Loss: 0.0538, Test Acc: 98.33%
Epoch [5/20] - Train Loss: 0.0376, Train Acc: 98.79% - Test Loss: 0.0324, Test Acc: 98.90%
Epoch [6/20] - Train Loss: 0.0314, Train Acc: 98.99% - Test Loss: 0.0330, Test Acc: 99.01%
Epoch [7/20] - Train Loss: 0.0280, Train Acc: 99.16% - Test Loss: 0.0283, Test Acc: 99.16%
Epoch [8/20] - Train Loss: 0.0261, Train Acc: 99.18% - Test Loss: 0.0244, Test Acc: 99.18%
Epoch [9/20] - Train Loss: 0.0229, Train Acc: 99.23% - Test Loss: 0.0292, Test Acc: 99.06%
Epoch [10/20] - Train Loss: 0.0193, Train Acc: 99.37% - Test Loss: 0.0237, Test Acc: 99.17%
Epoch [11/20] - Train Loss: 0.0173, Train Acc: 99.41% - Test Loss: 0.0377, Test Acc: 98.94%
Epoch [12/20] - Train Loss: 0.0159, Train Acc: 99.46% - Test Loss: 0.0262, Test Acc: 99.20%
Epoch [13/20] - Train Loss: 0.0148, Train Acc: 99.52% - Test Loss: 0.0268, Test Acc: 99.13%
Epoch [14/20] - Train Loss: 0.0123, Train Acc: 99.60% - Test Loss: 0.0271, Test Acc: 99.11%
Epoch [15/20] - Train Loss: 0.0122, Train Acc: 99.61% - Test Loss: 0.0384, Test Acc: 98.99%
Epoch [16/20] - Train Loss: 0.0108, Train Acc: 99.66% - Test Loss: 0.0286, Test Acc: 99.19%
Epoch [17/20] - Train Loss: 0.0096, Train Acc: 99.69% - Test Loss: 0.0371, Test Acc: 99.10%
Epoch [18/20] - Train Loss: 0.0099, Train Acc: 99.66% - Test Loss: 0.0321, Test Acc: 99.26%
Epoch [19/20] - Train Loss: 0.0081, Train Acc: 99.71% - Test Loss: 0.0327, Test Acc: 99.31%
Epoch [20/20] - Train Loss: 0.0063, Train Acc: 99.78% - Test Loss: 0.0296, Test Acc: 99.26%
```

### Results Summary
- **Total Parameters:** 88,584 (✅ Target: <100k)
- **Best Training Accuracy:** 99.78% (Epoch 20)
- **Best Test Accuracy:** 99.31% (Epoch 19) (✅ Target: >98.4%)
- **Final Test Accuracy:** 99.26%
- **Training Time:** 20 epochs (✅ Target: ≤20 epochs)

### Analysis

**Achievement:** The model successfully established a strong baseline with 99.31% test accuracy, significantly exceeding the 98.4% target and demonstrating effective feature learning with a well-structured CNN architecture.

**Criticism:** Clear overfitting observed with training accuracy (99.78%) substantially higher than test accuracy (99.26%), indicating the model memorizes training data rather than generalizing effectively to unseen examples.

**Improvement for Experiment 2:** Implement regularization techniques (Batch Normalization, Dropout) and reduce parameters to ~20k while adding data augmentation to bridge the train-test gap and achieve the 99.4% target more efficiently.

---

## Experiment 2 Results - Coming Soon
*Model architecture and results will be added after implementation*

---

## Experiment 3 Results - Coming Soon
*Model architecture and results will be added after implementation*

---

## Setup and Usage

```bash
# Clone the repository
git clone <repository-url>
cd tsai_assignment_6

# Install dependencies
pip install torch torchvision matplotlib torchsummary

# Run training
python train.py
```

## Files Description
- `model_1.py`: CNN_Model_1 architecture for Experiment 1
- `utils.py`: Utility functions for data loading, training, and evaluation
- `train.py`: Main training script
- `training_notebook.ipynb`: Jupyter notebook for experimentation
- `data/`: MNIST dataset storage
- `output/`: Training outputs and saved models