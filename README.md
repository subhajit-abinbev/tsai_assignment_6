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
Epoch [1/20] - Train Loss: 0.3123, Train Acc: 90.34% - Test Loss: 0.0948, Test Acc: 97.17%
Epoch [2/20] - Train Loss: 0.0877, Train Acc: 97.38% - Test Loss: 0.0665, Test Acc: 97.85%
Epoch [3/20] - Train Loss: 0.0594, Train Acc: 98.19% - Test Loss: 0.0496, Test Acc: 98.54%
Epoch [4/20] - Train Loss: 0.0487, Train Acc: 98.48% - Test Loss: 0.0447, Test Acc: 98.52%
Epoch [5/20] - Train Loss: 0.0388, Train Acc: 98.78% - Test Loss: 0.0443, Test Acc: 98.59%
Epoch [6/20] - Train Loss: 0.0342, Train Acc: 98.91% - Test Loss: 0.0364, Test Acc: 98.94%
Epoch [7/20] - Train Loss: 0.0276, Train Acc: 99.14% - Test Loss: 0.0418, Test Acc: 98.71%
Epoch [8/20] - Train Loss: 0.0257, Train Acc: 99.20% - Test Loss: 0.0415, Test Acc: 98.73%
Epoch [9/20] - Train Loss: 0.0224, Train Acc: 99.28% - Test Loss: 0.0336, Test Acc: 99.10%
Epoch [10/20] - Train Loss: 0.0195, Train Acc: 99.37% - Test Loss: 0.0400, Test Acc: 98.98%
Epoch [11/20] - Train Loss: 0.0179, Train Acc: 99.41% - Test Loss: 0.0321, Test Acc: 99.10%
Epoch [12/20] - Train Loss: 0.0166, Train Acc: 99.42% - Test Loss: 0.0345, Test Acc: 99.07%
Epoch [13/20] - Train Loss: 0.0153, Train Acc: 99.49% - Test Loss: 0.0423, Test Acc: 98.70%
Epoch [14/20] - Train Loss: 0.0134, Train Acc: 99.56% - Test Loss: 0.0363, Test Acc: 98.93%
Epoch [15/20] - Train Loss: 0.0112, Train Acc: 99.63% - Test Loss: 0.0370, Test Acc: 99.06%
Epoch [16/20] - Train Loss: 0.0106, Train Acc: 99.62% - Test Loss: 0.0547, Test Acc: 98.56%
Epoch [17/20] - Train Loss: 0.0106, Train Acc: 99.65% - Test Loss: 0.0341, Test Acc: 99.25%
Epoch [18/20] - Train Loss: 0.0086, Train Acc: 99.72% - Test Loss: 0.0352, Test Acc: 99.10%
Epoch [19/20] - Train Loss: 0.0084, Train Acc: 99.72% - Test Loss: 0.0409, Test Acc: 99.05%
Epoch [20/20] - Train Loss: 0.0095, Train Acc: 99.67% - Test Loss: 0.0430, Test Acc: 98.99%
```

### Results Summary
- **Total Parameters:** 88,584 (✅ Target: <100k)
- **Best Training Accuracy:** 99.72% (Epochs 18-19)
- **Best Test Accuracy:** 99.25% (Epoch 17) (✅ Target: >98.4%)
- **Final Test Accuracy:** 98.99%
- **Training Time:** 20 epochs (✅ Target: ≤20 epochs)

### Analysis

**Achievements:**
1. ✅ **Model Training:** Successfully built a working CNN that trains without errors
2. ✅ **Parameter Count:** 88,584 parameters well under the 100k target
3. ✅ **Accuracy Target:** Achieved 99.25% best test accuracy, significantly exceeding the 98.4% baseline
4. ✅ **Training Epochs:** Completed training within the 20-epoch limit
5. ✅ **Learning Behavior:** Model shows consistent learning with steady improvement

**Key Observations:**
1. **Strong Performance:** Achieves 98.4%+ test accuracy by epoch 3, surpassing target early
2. **Peak Performance:** Best test accuracy of 99.25% achieved at epoch 17
3. **Capacity Increase:** Model has 3x more parameters (88k vs 29k) providing better learning capacity
4. **Overfitting Indicators:** Training accuracy (99.72%) higher than final test accuracy (98.99%)
5. **Loss Fluctuation:** Test loss shows variability in later epochs while training loss decreases steadily

**Strengths:**
- Fast convergence to target accuracy
- Consistent high performance (98.5%+ from epoch 3 onwards)
- Good feature learning capability with deeper architecture
- Stable training without implementation errors

**Areas for Improvement (Next Experiments):**
1. **Regularization:** Add Batch Normalization and Dropout to reduce overfitting gap
2. **Parameter Efficiency:** Significantly reduce parameters (target <20k for Exp 2)
3. **Architecture Optimization:** Implement Global Average Pooling to replace large convolutions
4. **Data Augmentation:** Add image transformations for better generalization
5. **Learning Rate Scheduling:** Implement dynamic learning rate for better convergence
6. **Early Stopping:** Monitor validation loss to prevent overfitting in later epochs

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