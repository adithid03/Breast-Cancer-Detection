import streamlit as st
import joblib
import requests
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Breast Cancer Prediction App",page_icon="🩺",)

# Function to load the model from GitHub
@st.cache_resource
def load_model_from_github():
    url = "https://github.com/adithid03/Breast-Cancer-Detection/raw/refs/heads/main/adaboost_model.joblib"
    response = requests.get(url)
    if response.status_code == 200:
        model = joblib.load(BytesIO(response.content))
        return model
    else:
        st.error("Error: Unable to load the model from GitHub.")
        return None

# Load the model
model = load_model_from_github()

# Descriptive statistics for rotary slider ranges
variables = {
    "perimeter1": {"min": 43.79, "max": 188.5},
    "smoothness1": {"min": 0.05263, "max": 0.1634},
    "symmetry1": {"min": 0.106, "max": 0.304},
    "fractal_dimension1": {"min": 0.04996, "max": 0.09744},
    "texture2": {"min": 0.3602, "max": 4.885},
    "perimeter2": {"min": 0.757, "max": 21.98},
    "smoothness2": {"min": 0.001713, "max": 0.03113},
    "symmetry2": {"min": 0.007882, "max": 0.07895},
    "compactness3": {"min": 0.02729, "max": 1.058},
    "symmetry3": {"min": 0.1565, "max": 0.6638},
}

# Function to set background color
def set_background(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Set initial black background
set_background("#000000")  # Black background

# App title
st.title("Breast Cancer Detection (Malignant or Benign)")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Make Prediction", "Overview", "Developer Info"])

# Overview Page
if page == "Overview":
    st.subheader("Overview")
    st.write("""
    This app uses machine learning (AdaBoost classifier) to predict whether breast cancer is malignant or benign based on various features. The input features are normalized values for different variables that describe characteristics of a tumor.

    **Features in the dataset** include:
    - Perimeter
    - Smoothness
    - Symmetry
    - Fractal Dimension
    - Texture
    - Compactness

    The prediction is made based on these features, which are provided by the user via rotary sliders. The model has been trained using data and is deployed to give predictions in real-time.
    """)

    
        # Layout for images
    st.write("### ROC Curve")
    st.write("- Shows the performance of the AdaBoost classifier.")
    st.write("- X-axis: False Positive Rate (FPR).")
    st.write("- Y-axis: True Positive Rate (TPR).")
    st.write("- Curve closer to top-left = better model performance.")
    st.image("images/ROCcurve.png", use_container_width=True)

    st.write("### Classification Report")
    st.write("- Shows Precision, Recall, F1-Score, and Support for each class.")
    st.write("- **Precision**: Correctly predicted positive cases.")
    st.write("- **Recall**: Correctly identified actual positive cases.")
    st.write("- **F1-Score**: Harmonic mean of Precision and Recall.")
    st.image("images/ClassificationReport.png", use_container_width=True)

    st.write("### Heatmap of Correlation Matrix")
    st.write("- Visualizes relationships between features.")
    st.write("- Darker colors = stronger positive correlations.")
    st.write("- Helps identify redundant or influential features.")
    st.image("images/HeatmapofCorrelationMatrix.png", use_container_width=True)


# Developer Info Page
elif page == "Developer Info":
    st.subheader("Developer Information")
    st.write("""
    **Developer:** Durgam Adithi  
    **Project:** Breast Cancer Detection  
    **Model Used:** AdaBoost Classifier (Decision Tree as base estimator)  
    **Libraries Used:** 
    - pandas
    - scikit-learn
    - streamlit
    - requests
    - joblib
    - numpy
    - sklearn
    
    This project is part of an AI/ML research internship and is designed to predict the likelihood of breast cancer being malignant or benign.
    
    **Contact Information:**
    - **Contact:** +91 7995266892
    - **Email:** [adithidurgam22@gmail.com](mailto:adithidurgam22@gmail.com)
    - **LinkedIn:** [Durgam Adithi](https://www.linkedin.com/in/durgam-adithi-0267b6244/)
    - **GitHub:** [adithid03](https://github.com/adithid03)
    """)

# Make Prediction Page: User Input for Prediction
elif page == "Make Prediction":
    st.write(
        "Use the rotary sliders below to input normalized values for each variable. "
        "The range is normalized to [0, 1.5 × max] based on the descriptive statistics provided."
    )

    # Input feature sliders
    inputs = {}
    for var, stats in variables.items():
        max_val = stats["max"] * 1.5  # 1.5 times the maximum value
        min_val = stats["min"]
        inputs[var] = st.slider(
            label=f"{var} (Normalized)",
            min_value=0.0,
            max_value=max_val,
            value=min_val,
            step=max_val / 100,  # Adjust precision as needed
            format="%.4f",
        )

    # Prepare input for prediction
    if model:
        input_features = np.array(list(inputs.values())).reshape(1, -1)

        # Prediction button
        if st.button("Predict"):
            prediction = model.predict(input_features)[0]
            if prediction == 0:
                set_background("#013220")  # Dark green for benign
                st.success("The prediction is: 🎉 Benign (0)")
            else:
                set_background("#8B0000")  # Dark red for malignant
                st.error("The prediction is: ⚠️ Malignant (1)")
            st.markdown("""
                **Medical Disclaimer:**  
                - The predictions made by this app are based on machine learning models trained on historical data. 
                - While the model aims to provide accurate predictions, it should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
                - Always consult a qualified healthcare professional for any medical concerns.
            """)
    else:
        st.error("Model not loaded. Please check the GitHub URL.")
