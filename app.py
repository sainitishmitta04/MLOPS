import numpy as np
import streamlit as st
from pickle import load
import warnings
warnings.filterwarnings("ignore")

sc = load(open('models/standard_scaler.pkl','rb'))
svm = load(open('models/svm_model.pkl','rb'))

st.title(":red[Breast Cancer Detection]")

import streamlit as st


mean_radius = st.number_input('Enter the Mean Radius Value')
mean_texture = st.number_input('Enter the Mean Texture Value')
mean_perimeter = st.number_input('Enter the Mean Perimeter Value')
mean_area = st.number_input('Enter the Mean Area Value')
mean_smoothness = st.number_input('Enter the Mean Smoothness Value')
mean_compactness = st.number_input('Enter the Mean Compactness Value')
mean_concavity = st.number_input('Enter the Mean Concavity Value')
mean_concave_points = st.number_input('Enter the Mean Concave Points Value')
mean_symmetry = st.number_input('Enter the Mean Symmetry Value')
mean_fractal_dimension = st.number_input('Enter the Mean Fractal Dimension Value')
radius_error = st.number_input('Enter the Radius Error Value')
texture_error = st.number_input('Enter the Texture Error Value')
perimeter_error = st.number_input('Enter the Perimeter Error Value')
area_error = st.number_input('Enter the Area Error Value')
smoothness_error = st.number_input('Enter the Smoothness Error Value')
compactness_error = st.number_input('Enter the Compactness Error Value')
concavity_error = st.number_input('Enter the Concavity Error Value')
concave_points_error = st.number_input('Enter the Concave Points Error Value')
symmetry_error = st.number_input('Enter the Symmetry Error Value')
fractal_dimension_error = st.number_input('Enter the Fractal Dimension Error Value')
worst_radius = st.number_input('Enter the Worst Radius Value')
worst_texture = st.number_input('Enter the Worst Texture Value')
worst_perimeter = st.number_input('Enter the Worst Perimeter Value')
worst_area = st.number_input('Enter the Worst Area Value')
worst_smoothness = st.number_input('Enter the Worst Smoothness Value')
worst_compactness = st.number_input('Enter the Worst Compactness Value')
worst_concavity = st.number_input('Enter the Worst Concavity Value')
worst_concave_points = st.number_input('Enter the Worst Concave Points Value')
worst_symmetry = st.number_input('Enter the Worst Symmetry Value')
worst_fractal_dimension = st.number_input('Enter the Worst Fractal Dimension Value')

# Create a list of column names
columns = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness,
mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error,
perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error,
symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area,
worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry,
worst_fractal_dimension]


# input = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

if st.button('Predict'):
    query_point = np.array(columns)
    query_point = query_point.reshape(1,-1)
    query_point_transformed = sc.transform(query_point)
    prediction = svm.predict(query_point_transformed)
    if (prediction[0] == 0):
        st.error('The Breast cancer is Malignant')
        st.subheader(':blue[Precautions for Malignant breast cancer:]')
        bullet_points = ["Perform regular breast self-exams",
                         "Maintain a healthy lifestyle",
                         "Limit alcohol consumption",
                         "Know your family history",
                         "Get regular mammograms and other screening tests"]

        for point in bullet_points:
            st.write(f"- {point}")

        st.subheader(":green[Tips for managing malignant breast cancer:]")
        malignant_points = ["Educate yourself about your diagnosis",
                            "Work with a team of healthcare professionals",
                            "Communicate with your loved ones",
                            "Manage treatment side effects",
                            "Take care of yourself during and after treatment"]
        for point in malignant_points:
            st.write(f"- {point}")

    else:
        st.error('The Breast Cancer is Benign')
        st.subheader(':blue[Precautions for Benign breast cancer:]')
        bullet_points = ["Perform regular breast self-exams",
                         "Maintain a healthy lifestyle",
                         "Limit alcohol consumption",
                         "Know your family history",
                         "Get regular breast exams"]

        for point in bullet_points:
            st.write(f"- {point}")

        st.subheader(":green[Tips for managing benign breast cancer:]")
        benign_points = ["Follow your healthcare provider's recommendations",
                         "Be aware of any changes",
                         "Manage any symptoms",
                         "Seek support",
                         "Consider surgery if necessary"]
        for point in benign_points:
            st.write(f"- {point}")

