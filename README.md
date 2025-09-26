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
- Make the model lighter (reduce parameters closer to ~30k)
- Add Batch Normalization, and Dropout to improve generalization
- Experiment with Global Average Pooling to reduce parameters
- Target: ≥ 99.4% test accuracy within 20 epochs, parameters < 30k

### Experiment 3 – Final Optimized Model
**Targets:**
- Achieve the required benchmark: ≥ 99.4% test accuracy consistently in the last few epochs
- Train within ≤ 15 epochs
- Keep total parameters ≤ 8,000
- Use best practices: proper skeleton, BN, Dropout, proper MaxPooling placement, GAP, data augmentation, lighter architecture, and tuned learning rate schedule
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

**Improvement for Experiment 2:** Implement regularization techniques (Batch Normalization, Dropout) and reduce parameters to ~30k by adding GAP bridge the train-test gap and achieve the 99.4% target more efficiently.

---

## Experiment 2 Results - CNN_Model_2 ✅

### Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
       BatchNorm2d-2           [-1, 16, 26, 26]              32
            Conv2d-3           [-1, 16, 24, 24]           2,304
       BatchNorm2d-4           [-1, 16, 24, 24]              32
            Conv2d-5           [-1, 32, 22, 22]           4,608
       BatchNorm2d-6           [-1, 32, 22, 22]              64
         MaxPool2d-7           [-1, 32, 11, 11]               0
            Conv2d-8           [-1, 16, 11, 11]             512
       BatchNorm2d-9           [-1, 16, 11, 11]              32
           Conv2d-10             [-1, 16, 9, 9]           2,304
      BatchNorm2d-11             [-1, 16, 9, 9]              32
          Dropout-12             [-1, 16, 9, 9]               0
           Conv2d-13             [-1, 32, 7, 7]           4,608
      BatchNorm2d-14             [-1, 32, 7, 7]              64
           Conv2d-15             [-1, 40, 5, 5]          11,520
      BatchNorm2d-16             [-1, 40, 5, 5]              80
           Conv2d-17             [-1, 10, 5, 5]             400
      BatchNorm2d-18             [-1, 10, 5, 5]              20
AdaptiveAvgPool2d-19             [-1, 10, 1, 1]               0
================================================================
Total params: 26,756
Trainable params: 26,756
Non-trainable params: 0
----------------------------------------------------------------
```

### Training Logs
```
Epoch [1/20] - Train Loss: 0.6114, Train Acc: 90.95% - Test Loss: 0.1575, Test Acc: 97.92%
Epoch [2/20] - Train Loss: 0.1345, Train Acc: 98.04% - Test Loss: 0.0876, Test Acc: 98.56%
Epoch [3/20] - Train Loss: 0.0877, Train Acc: 98.53% - Test Loss: 0.0846, Test Acc: 98.49%
Epoch [4/20] - Train Loss: 0.0675, Train Acc: 98.81% - Test Loss: 0.0707, Test Acc: 98.22%
Epoch [5/20] - Train Loss: 0.0568, Train Acc: 98.89% - Test Loss: 0.0579, Test Acc: 98.79%
Epoch [6/20] - Train Loss: 0.0488, Train Acc: 99.07% - Test Loss: 0.0418, Test Acc: 99.10%
Epoch [7/20] - Train Loss: 0.0424, Train Acc: 99.18% - Test Loss: 0.0439, Test Acc: 98.92%
Epoch [8/20] - Train Loss: 0.0377, Train Acc: 99.25% - Test Loss: 0.0323, Test Acc: 99.27%
Epoch [9/20] - Train Loss: 0.0341, Train Acc: 99.33% - Test Loss: 0.0494, Test Acc: 98.70%
Epoch [10/20] - Train Loss: 0.0321, Train Acc: 99.35% - Test Loss: 0.0287, Test Acc: 99.36%
Epoch [11/20] - Train Loss: 0.0291, Train Acc: 99.41% - Test Loss: 0.0292, Test Acc: 99.26%
Epoch [12/20] - Train Loss: 0.0271, Train Acc: 99.45% - Test Loss: 0.0301, Test Acc: 99.32%
Epoch [13/20] - Train Loss: 0.0249, Train Acc: 99.51% - Test Loss: 0.0320, Test Acc: 99.26%
Epoch [14/20] - Train Loss: 0.0243, Train Acc: 99.51% - Test Loss: 0.0256, Test Acc: 99.38%
Epoch [15/20] - Train Loss: 0.0220, Train Acc: 99.59% - Test Loss: 0.0276, Test Acc: 99.19%
Epoch [16/20] - Train Loss: 0.0207, Train Acc: 99.62% - Test Loss: 0.0222, Test Acc: 99.42%
Epoch [17/20] - Train Loss: 0.0195, Train Acc: 99.64% - Test Loss: 0.0266, Test Acc: 99.34%
Epoch [18/20] - Train Loss: 0.0184, Train Acc: 99.67% - Test Loss: 0.0237, Test Acc: 99.36%
Epoch [19/20] - Train Loss: 0.0188, Train Acc: 99.65% - Test Loss: 0.0234, Test Acc: 99.44%
Epoch [20/20] - Train Loss: 0.0169, Train Acc: 99.71% - Test Loss: 0.0235, Test Acc: 99.41%
```

### Results Summary
- **Total Parameters:** 26,756 (✅ Target: <30k - **Need optimization**)
- **Best Training Accuracy:** 99.71% (Epoch 20)
- **Best Test Accuracy:** 99.44% (Epoch 19) (✅ Target: ≥99.4%)
- **Final Test Accuracy:** 99.41%
- **Training Time:** 20 epochs (✅ Target: ≤20 epochs)

### Analysis

**Achievement:** The model successfully achieved the 99.4% test accuracy target (99.44% best, 99.41% final) with effective regularization through Batch Normalization and Dropout, demonstrating excellent generalization with minimal overfitting gap.

**Criticism:** Parameter count at 26,756 significantly exceeds the 30k target, requiring further optimization through architectural improvements and more efficient layer designs to meet the parameter constraint.

**Improvement for Experiment 3:** Reduce parameters through more efficient architecture design, implement advanced regularization techniques, add data augmentation, and optimize learning rate schedule to achieve ≥99.4% accuracy consistently within 15 epochs and ≤8k parameters.

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
- `model_2.py`: CNN_Model_2 architecture for Experiment 2 with BatchNorm and Dropout
- `utils.py`: Utility functions for data loading, training, and evaluation
- `train.py`: Main training script
- `training_notebook.ipynb`: Jupyter notebook for experimentation
- `data/`: MNIST dataset storage
- `output/`: Training outputs and saved models