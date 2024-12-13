# Breast Cancer Detection App using AdaBoost Classifier
## Introduction
Breast cancer is one of the most common cancers affecting women worldwide. It occurs when cells in the breast grow uncontrollably, forming a tumor that can sometimes spread to other parts of the body.
It is the most common cancer among women and can also occur in men, although rarely.

## Description
This application leverages machine learning, specifically the AdaBoost classifier, to predict whether a breast tumor is malignant or benign based on various features. It provides real-time predictions from user inputs through an intuitive web interface built using Streamlit. The application also offers insights into model performance with detailed visualizations and metrics.

## Features
- Real-time prediction of breast cancer (malignant/benign) based on user inputs.
- Uses an AdaBoost classifier for prediction.
- Visual representation of the classifier's performance with ROC curves.
- Detailed classification report (Precision, Recall, F1-Score).
- Heatmap to visualize correlations between different features.

## Dataset
- **Source**: The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset.
- Target Variable:
   - Malignant (1): Cancerous tumors.
   - Benign (0): Non-cancerous tumors.

## Technologies Used
- Python 3.x
- Streamlit
- scikit-learn
- joblib
- Matplotlib/Seaborn
- Pandas/Numpy

## Installation Instructions
### Prerequisites
- Python 3.x
- pip

### Installation
1. Clone this repository:
    ```
    git clone https://github.com/adithid03/Breast-Cancer-Detection.git
    ```
2. Navigate to the project directory:
    ```
    cd Breast-Cancer-Detection
    ```
3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```
    streamlit run app.py
    ```

## How It Works
The app takes normalized input values through sliders for features such as perimeter, smoothness, and symmetry of breast tumors. Based on these inputs, the AdaBoost classifier predicts whether the tumor is benign or malignant. The classifier's performance is visualized with an ROC curve, and additional statistics like precision, recall, and F1-score are displayed in the classification report.

## Usage Instructions
1. Launch the app with Streamlit.
2. Navigate to the "Make Prediction" page.
3. Use the sliders to input normalized values for each feature.
4. Click "Predict" to see if the tumor is malignant or benign.
5. Review the classification report and performance metrics in "Overview" page.

## Model Performance
- Accuracy: 92%
- Precision: 90%
- Recall: 94%
- F1-Score: 92%

## Medical Disclaimer
- The predictions made by this app are based on machine learning models trained on historical data. 
- While the model aims to provide accurate predictions, it should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
- Always consult a qualified healthcare professional for any medical concerns.