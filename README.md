# ISE364HW4
Penguins Regression and CIFAR-10 Image Classification with PyTorch

This repository contains two complete machine learning projects implemented in PyTorch. The first focuses on regression using the Penguins dataset, and the second performs image classification using the CIFAR-10 dataset. Together, they demonstrate how to handle tabular and image data in deep learning workflows.

Project 1: Penguins Dataset (Regression)

Objective
Predict the body mass of penguins using their morphological measurements such as bill length, bill depth, flipper length, and categorical variables like species, island, and sex.

Steps

Data Loading and Cleaning
The dataset is imported from the Seaborn library and cleaned by removing missing values. Features are separated into input (X) and target (y), where the target variable is body_mass_g.

Feature Engineering
Categorical variables are one-hot encoded, and numerical features are standardized using StandardScaler to ensure stable model training.

PyTorch Dataset and DataLoader
A custom Dataset class converts pandas DataFrames into PyTorch tensors. DataLoaders are created for batching and shuffling during training.

Model Development

A fully connected neural network (FCNN) is implemented with three hidden layers using ReLU activation and dropout for regularization.

A linear regression model is also built for comparison.

Training
The FCNN is trained using the Adam optimizer and Mean Squared Error (MSE) loss. The data is split into training (70%) and testing (30%) sets using torch.utils.data.random_split. The model is trained for 100 epochs, and loss curves are plotted to track convergence.

Evaluation
Performance is evaluated using Mean Absolute Error (MAE) and R² Score.
The linear regression model achieved an MAE of approximately 223 g and an R² of 0.83, outperforming the FCNN (MAE ≈ 317 g, R² ≈ 0.69), suggesting that the relationship between features and body mass is mostly linear.

Results Summary
The project illustrates that for simple structured data, a linear regression model can outperform more complex networks due to lower variance and better generalization on small datasets.

Project 2: CIFAR-10 Dataset (Image Classification)

Objective
Classify 32x32 color images into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

Steps

Data Loading and Normalization
The dataset is loaded using torchvision.datasets.CIFAR10 and transformed into tensors. Normalization is applied to scale RGB pixel values to [-1, 1]. DataLoaders are prepared for both training and testing with a batch size of 64.

Data Visualization
Random samples of 12 images are displayed with their corresponding class labels to verify dataset integrity and preprocessing correctness.

CNN Model Architecture

Base CNN: Three convolutional layers followed by max pooling, dropout, and two fully connected layers.

Improved CNN: Adds Batch Normalization and an extra convolutional block for better generalization.

Training
Models are trained using the CrossEntropyLoss function and the Adam optimizer for 10 epochs. The training process includes tracking both training loss and test accuracy at each epoch.

Evaluation
The trained models are evaluated using accuracy as the main metric. Visualizations include:

True and predicted labels for 12 test images

Training loss and test accuracy plots

Model Comparisons
Three models are implemented and compared:

Logistic Regression (baseline, ~40% accuracy)

Base CNN (~70% accuracy)

Improved CNN (~78% accuracy)

Results Summary
The improved CNN achieved the highest accuracy, demonstrating the benefit of deeper architectures, batch normalization, and dropout. CNNs outperform logistic regression because they learn hierarchical spatial features crucial for visual recognition.

How to Run

Install dependencies:
pip install torch torchvision seaborn pandas matplotlib scikit-learn numpy

Open the Jupyter notebook (HW4.ipynb) and run each section in order.

GPU acceleration (if available) will significantly speed up the CNN training on CIFAR-10.