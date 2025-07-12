# 🤖 Machine Learning Projects Portfolio

Welcome to my comprehensive collection of Machine Learning projects! This repository showcases my journey through various ML concepts, algorithms, and real-world applications. Each project demonstrates different aspects of machine learning, from fundamental algorithms to advanced deep learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This portfolio contains a diverse range of machine learning projects covering:
- **Supervised Learning**: Classification and Regression problems
- **Deep Learning**: Neural networks for image recognition
- **Data Analysis**: Exploratory data analysis and visualization
- **Computer Vision**: Image processing and classification
- **Natural Language Processing**: Text analysis and processing
- **Model Evaluation**: Performance metrics and evaluation techniques

## 🚀 Projects

### 1. **Autism Spectrum Disorder (ASD) Detection** 
📁 `ASD.ipynb` | 🔍 **Classification, Healthcare ML**

A comprehensive machine learning project for early detection of Autism Spectrum Disorder in toddlers. This project uses multiple algorithms to predict ASD traits based on behavioral and demographic features.

**Key Features:**
- Multiple ML algorithms (Logistic Regression, Random Forest, SVM, Neural Networks)
- Feature engineering and preprocessing
- Model comparison and evaluation
- Healthcare-focused application

**Technologies:** Scikit-learn, TensorFlow, Pandas, NumPy, Matplotlib, Seaborn

---

### 2. **Iris Dataset Analysis & Classification**
📁 `iris-dataset.ipynb` | 🌸 **Classification, Data Visualization**

Classic machine learning project using the famous Iris dataset. Demonstrates fundamental ML concepts with beautiful visualizations.

**Key Features:**
- Exploratory data analysis
- Correlation analysis and heatmaps
- Distribution plots and scatter plots
- Feature engineering with dummy variables

**Technologies:** Pandas, Matplotlib, Seaborn, Scikit-learn

---

### 3. **Linear Regression - House Price Prediction**
📁 `linear-regression.ipynb` | 🏠 **Regression, Real Estate ML**

Predicts house prices using linear regression based on living area and other features. Includes model evaluation and visualization.

**Key Features:**
- Simple and multiple linear regression
- Model performance evaluation (MSE, R²)
- Data visualization with regression lines
- Real-world application in real estate

**Technologies:** Scikit-learn, Pandas, NumPy, Matplotlib

---

### 4. **Decision Tree Classification**
📁 `decision-tree.ipynb` | 🌳 **Classification, Tree-based ML**

Implementation of decision tree classifier with comprehensive model evaluation and tree visualization.

**Key Features:**
- Decision tree implementation
- Model accuracy assessment
- Classification report generation
- Tree structure visualization
- Perfect accuracy demonstration on Iris dataset

**Technologies:** Scikit-learn, Pandas, Matplotlib

---

### 5. **Phishing Website Detection**
📁 `phishing.ipynb` | 🛡️ **Classification, Cybersecurity ML**

Advanced machine learning system for detecting phishing websites using URL features and Random Forest algorithm.

**Key Features:**
- Feature extraction from URLs
- Random Forest classification
- Model persistence and deployment
- Interactive prediction interface
- Cybersecurity application

**Technologies:** Scikit-learn, Pandas, NumPy, Joblib, Gradio

---

### 6. **MNIST Digit Recognition**
📁 `mnsit-pred.ipynb` | 🔢 **Deep Learning, Computer Vision**

Neural network implementation for handwritten digit recognition using the MNIST dataset.

**Key Features:**
- Deep neural network architecture
- Image preprocessing and normalization
- Training visualization (accuracy/loss plots)
- Real-time prediction demonstration
- Computer vision application

**Technologies:** TensorFlow, Keras, NumPy, Matplotlib

---

### 7. **MNIST Classification Report**
📁 `mnsit-classification-report.ipynb` | 📊 **Model Evaluation, Deep Learning**

Comprehensive evaluation and analysis of MNIST classification model performance.

**Key Features:**
- Detailed classification metrics
- Confusion matrix analysis
- Model performance insights
- Error analysis and visualization

**Technologies:** TensorFlow, Keras, Scikit-learn, Matplotlib, Seaborn

---

### 8. **PCA Visualization**
📁 `pca-visualization.ipynb` | 📈 **Dimensionality Reduction, Data Visualization**

Principal Component Analysis implementation with comprehensive visualization techniques.

**Key Features:**
- PCA implementation and analysis
- Dimensionality reduction visualization
- Explained variance analysis
- Data transformation insights

**Technologies:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

### 9. **Confusion Matrix Analysis**
📁 `confusion-matrix.ipynb` | 📊 **Model Evaluation, Metrics**

Detailed analysis of model performance using confusion matrices and related metrics.

**Key Features:**
- Confusion matrix implementation
- Precision, Recall, F1-score analysis
- Model comparison techniques
- Performance visualization

**Technologies:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

### 10. **Linear Regression Evaluation**
📁 `lr-eval.ipynb` | 📊 **Regression Evaluation, Model Assessment**

Comprehensive evaluation of linear regression models with various metrics and visualizations.

**Key Features:**
- Multiple evaluation metrics
- Residual analysis
- Model diagnostics
- Performance comparison

**Technologies:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

### 11. **Water Level Detection with OpenCV**
📁 `waterlog-opencv.ipynb` | 🌊 **Computer Vision, Image Processing**

Computer vision project for water level detection using OpenCV and image processing techniques.

**Key Features:**
- Image processing with OpenCV
- Water level detection algorithms
- Real-time processing capabilities
- Computer vision applications

**Technologies:** OpenCV, NumPy, Matplotlib

---

## 🛠️ Technologies Used

### Core Machine Learning
- **Scikit-learn**: Classification, regression, clustering, model evaluation
- **TensorFlow/Keras**: Deep learning and neural networks
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis

### Data Visualization
- **Matplotlib**: Basic plotting and visualization
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive visualizations

### Computer Vision
- **OpenCV**: Image processing and computer vision
- **PIL/Pillow**: Image manipulation

### Model Deployment
- **Joblib**: Model serialization and persistence
- **Gradio**: Interactive web interfaces

### Development Environment
- **Jupyter Notebooks**: Interactive development and documentation
- **Google Colab**: Cloud-based development environment

## 🚀 Getting Started

### Prerequisites

```bash
# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python joblib gradio
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ml-projects-portfolio.git
   cd ml-projects-portfolio
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

### Running Projects

Each project is contained in its own Jupyter notebook. To run a specific project:

1. Open the desired `.ipynb` file
2. Follow the instructions in the notebook
3. Upload required datasets when prompted
4. Execute cells sequentially

## 📁 Project Structure

```
ml/
├── ASD.ipynb                           # Autism detection project
├── iris-dataset.ipynb                  # Iris classification
├── linear-regression.ipynb             # House price prediction
├── decision-tree.ipynb                 # Decision tree classification
├── phishing.ipynb                      # Phishing detection
├── mnsit-pred.ipynb                   # MNIST digit recognition
├── mnsit-classification-report.ipynb   # MNIST evaluation
├── pca-visualization.ipynb            # PCA analysis
├── confusion-matrix.ipynb              # Model evaluation
├── lr-eval.ipynb                      # Linear regression evaluation
└── waterlog-opencv.ipynb              # Water level detection
```

## 📊 Key Learning Outcomes

Through these projects, I've gained expertise in:

- **Supervised Learning**: Classification and regression techniques
- **Deep Learning**: Neural network architectures and training
- **Data Preprocessing**: Feature engineering, scaling, encoding
- **Model Evaluation**: Metrics, cross-validation, hyperparameter tuning
- **Data Visualization**: Statistical plots, correlation analysis
- **Computer Vision**: Image processing and recognition
- **Real-world Applications**: Healthcare, cybersecurity, real estate

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)
- **Email**: your.email@example.com

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the Machine Learning community

</div> 
