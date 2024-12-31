# Image Classification Model Results
<p align="center">
    <!--PANDAS-->
      <img src="Images\Pandas_logo.svg.png"  width=15% alt="Pandas">
   <!--PLUS SIGN-->
          <img src="Images\plus_sign.svg" width=7% alt="Plus">
    <!--NUMPY-->
      <img src="Images\NumPy_logo_2020.svg.png"  width=15% alt="Numpy">
   <!--PLUS SIGN-->
          <img src="Images\plus_sign.svg" width=7% alt="Plus">
    <!--SCIKITLEARN-->
      <img src="Images\Scikit_learn_logo_small.svg.png"  width=10% alt="Sci-kit Learn">
   <!--PLUS SIGN-->
          <img src="Images\plus_sign.svg" width=7% alt="Plus">
     <!--TENSORFLOW-->
          <img src="Images\TensorFlow_logo.svg.png" width=15% alt="Tensorflow">

## About the Dataset

<p align="center">
    <!--FEATUREEXTRACIONS-->
      <img src="Images\sample_images.png"  width=90% alt="FeatureExtractions">
  
This dataset comprises of 25,000 images, each with a resolution of 150x150 pixels, divided into six categories: Buildings, Forest, Glacier, Mountain, Sea, and Street. The data is organized into separate zip files for training, testing, and prediction, with around 14,000 images in the training set, 3,000 in the test set, and 7,000 for prediction. In the training set, each feature has roughly 2,300 examples. In the test set, each feature has 500 examples.

## Feature Engineering Approaches

1. **Raw Images**
  - Original RGB images used as baseline input
  - Minimal preprocessing with standard normalization
  - Direct pixel values as features

2. **HSV Color Space**
  - Conversion from RGB to HSV color space
  - Better representation of color information
  - More robust to lighting variations

3. **Histogram of Oriented Gradients (HOG)**
  - Extraction of gradient-based features
  - Captures shape and edge information
  - Particularly effective for structural elements

# Model Performance Comparison

## Traditional Machine Learning Models

### Perceptron Results
| Feature Extraction | Accuracy | Training Time | Inference/Image (s) |
|-------------------|----------|---------------|-------------------|
| Raw               | 27%      | 24m 49.5s    | 0.4965           |
| HSV               | 34%      | 13m 35.1s    | 0.2717           |
| HOG               | 43%      | 24m 1.6s     | 0.4805           |

### Logistic Regression Results
| Feature Extraction | Accuracy | Training Time | Inference/Image (s) |
|-------------------|----------|---------------|-------------------|
| Raw               | 40%      | 29m 16.4s    | 0.5855           |
| HSV               | 44%      | 55m 34.5s    | 1.1115           |
| HOG               | 43%      | 30m 55.6s    | 0.6185           |

### SVM Results
| Feature Extraction (PCA) | Accuracy | Training Time | Inference/Image (s) |
|-------------------------|----------|---------------|-------------------|
| HSV (PCA=50)           | 63%      | 10m 12.9s    | 0.2043           |
| HOG (PCA=100)          | 63%      | 8m 7.6s      | 0.1625           |
| ResNet (PCA=128)       | 71%      | 4m 54.8s     | 0.0983           |

## Deep Learning Models

### CNN Results
| Feature Extraction  | Accuracy | Training Time | Inference/Image (s) |
|--------------------|----------|---------------|-------------------|
| Naïve              | 79.9%    | 15m 45.8s    | 0.0003           |
| HSV                | 74.0%    | 13m 9.2s     | 0.0002           |
| HOG                | 79.1%    | 14m 37.8s    | 0.0003           |
| Multimodal (PCA)   | 80.6%    | 22.1s            | 0.0003           |

### ResNet Results
| Feature Extraction  | Accuracy | Training Time | Inference/Image (s) |
|--------------------|----------|---------------|-------------------|
| Naïve              | 69.7%    | 69m 46.8s    | 0.0002           |
| HSV                | 36.0%    | 69m 13.0s    | 0.0001           |
| HOG                | 65.0%    | 65m 47.7s    | 0.0002           |
| Multimodal (PCA)   | 79.9%    | 41m 45.9s    | 0.0003           |

</div>

## Key Findings

1. **Best Performance**: The Multimodal CNN achieved the highest accuracy at 80.6% with relatively fast classification time.

2. **Model Progression**:
   - Basic linear models (Perceptron, Logistic Regression) provided baseline performance
   - SVM with Resnet showed significant improvement
   - Deep learning models (CNN, ResNet50) achieved the best results overall

3. **Feature Engineering Impact**:
   - HOG features consistently improved performance across models
   - Multimodal approaches (combining raw images with engineered features) outperformed naive implementations

4. **Efficiency vs. Accuracy**:
   - SVM offers the best trade-off between training time and performance
   - ResNet50 required significantly more training time without proportional accuracy gains
   - Multimodal approaches generally improved accuracy with minimal additional computational cost

5. **Common Challenges**:
   - Difficulty distinguishing between visually similar categories (e.g., glacier vs. mountain)
   - Urban scene classification showed consistent improvement across models
   - Natural scenes posed more classification challenges due to subtle feature differences