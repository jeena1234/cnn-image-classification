# CNN Image Classification: Cats, Dogs, and Pandas

A computer vision project using TensorFlow/Keras to classify images into three categories: **cats, dogs, and pandas**.

The project explores the complete image-classification workflow, including dataset inspection, preprocessing, CNN design, data augmentation, optimizer comparison, model evaluation, and error analysis.

## Project Overview

The dataset contains **3,000 images** distributed equally across three classes:

* Cats
* Dogs
* Pandas

Before training, the images were inspected for corrupted files, image dimensions, and color formats. The dataset was then divided into balanced training, validation, and test sets.

| Dataset Split | Images per Class | Total Images |
| ------------- | ---------------: | -----------: |
| Training      |              700 |        2,100 |
| Validation    |              100 |          300 |
| Test          |              200 |          600 |

## Workflow

The project includes:

1. Dataset inspection and corrupted-image validation
2. Image dimension and RGB/grayscale analysis
3. Train/validation/test dataset creation
4. TensorFlow image preprocessing pipeline
5. Baseline CNN development
6. Data augmentation experiments
7. Deeper VGG-style CNN experiment
8. Adam and RMSprop optimizer comparison
9. Training and validation curve analysis
10. Test-set evaluation and confusion-matrix analysis

## Model Experiments

Four CNN configurations were evaluated.

| Model                   | Test Accuracy |  Test Loss |
| ----------------------- | ------------: | ---------: |
| **Baseline CNN + Adam** |    **73.83%** | **0.5796** |
| CNN + RMSprop           |        72.83% |     0.5909 |
| CNN + Data Augmentation |        72.83% |     0.6764 |
| VGG-style CNN           |        68.33% |     0.6955 |

## Key Findings

The **baseline CNN using the Adam optimizer achieved the best overall performance**, reaching **73.83% test accuracy** with the lowest test loss of **0.5796**.

The RMSprop model performed similarly but did not outperform Adam. Data augmentation also did not improve test accuracy for the augmentation settings used in this experiment.

Increasing network depth with the VGG-style architecture resulted in lower test performance. This experiment demonstrated that a more complex model does not necessarily provide better generalization, particularly for a relatively small dataset.

Overall, the simpler baseline CNN provided the best balance of model complexity and test performance among the approaches evaluated.

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Pillow
* scikit-learn
* Jupyter Notebook

## Repository Structure

```text
cnn-image-classification/
│
├── notebooks/
│   └── cnn_image_classification.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

The image dataset is not stored in this repository.

## Running the Project

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then open:

```text
notebooks/cnn_image_classification.ipynb
```

The notebook contains the full workflow, model experiments, evaluation results, and visualizations.

## What I Learned

This project provided hands-on experience with building and evaluating convolutional neural networks for multiclass image classification. It also demonstrated the importance of comparing model architectures experimentally rather than assuming that additional depth, augmentation, or a different optimizer will automatically improve performance.

