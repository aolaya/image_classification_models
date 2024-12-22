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
          <img src="Images\TensorFlow_logo.svg.png" width=10% alt="Tensorflow">

## About the Dataset

<p align="center">
    <!--FEATUREEXTRACIONS-->
      <img src="Images\sample_images.png"  width=90% alt="FeatureExtractions">
  
#### This dataset comprises of 25,000 images, each with a resolution of 150x150 pixels, divided into six categories: Buildings, Forest, Glacier, Mountain, Sea, and Street. The data is organized into separate zip files for training, testing, and prediction, with around 14,000 images in the training set, 3,000 in the test set, and 7,000 for prediction. In the training set, each feature has roughly 2,300 examples. In the test set, each feature has 500 examples.

## Model Performance Comparison

<div align="center">

### Traditional Machine Learning Models
| Model Type | Feature Type | Accuracy | Training Time | Inference Time/Image |
|------------|-------------|-----------|---------------|---------------------|
| Perceptron | HOG | **43.0%** | 9m 3.2s | 0.001633s |
| Perceptron | Original | 40.0% | 7m 29s | 0.000467s |
| Perceptron | HSV | 34.0% | 7m 14.6s | 0.000167s |

### Logistic Regression Models
| Model Type | Feature Type | Accuracy | Training Time | Inference Time/Image |
|------------|-------------|-----------|---------------|---------------------|
| Logistic Regression | HOG | **49.0%** | 23m 30.0s | 0.00617s |
| Logistic Regression | HSV | 42.0% | 17m 3.0s | 0.00390s |
| Logistic Regression | Original | 41.0% | 14m 7.7s | 0.00073s |

### Support Vector Machines (SVM)
| Model Type | Feature Type | Accuracy | Training Time | Inference Time/Image |
|------------|-------------|-----------|---------------|---------------------|
| SVM | HOG + PCA(100) | **68.00%** | 1.84s | 0.002034s |
| SVM | ResNet + PCA(128) | 62.50% | 2.02s | 0.017395s |
| SVM | HSV + PCA(50) | 53.00% | 1.53s | 0.015857s |

### Deep Learning Models
| Model Type | Feature Type | Accuracy | Training Time | Inference Time/Image |
|------------|-------------|-----------|---------------|---------------------|
| CNN | Multimodal | **81.93%** | 13m 37.5s | 0.001954s |
| ResNet50 | Multimodal | 79.80% | 58m 24.6s | 0.017395s |
| CNN | Naive | 79.70% | 13m 33.6s | 0.002034s |
| ResNet50 | Naive | 69.60% | 54m 47.4s | 0.015857s |

</div>

## Key Findings

1. **Best Performance**: The Multimodal CNN achieved the highest accuracy at 81.93% with relatively fast classification time.

2. **Model Progression**:
   - Basic linear models (Perceptron, Logistic Regression) provided baseline performance
   - SVM with HOG features showed significant improvement
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

## Technologies Used
- Machine Learning: Scikit-learn, TensorFlow
- Feature Extraction: HOG, HSV, ResNet features
- Deep Learning: CNN, ResNet50
- Data Processing: NumPy, Pandas