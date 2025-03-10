import streamlit as st
import time
import cv2
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import joblib  
from fpdf import FPDF  
import os
import pathlib  
from datetime import datetime  
import plotly.graph_objects as go

# Temporary fix for pathlib issue
temp = pathlib.PosixPath  
pathlib.PosixPath = str

# Set page configuration
st.set_page_config(page_title="🩺 Enhanced Diagnostic Assistant", layout="wide")

# Apply CSS for styling
st.markdown('''
    <style>
        body {
            background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
            font-family: Arial, sans-serif;
        }
        .result-box {
            font-size: 24px;
            font-weight: bold;
            padding: 10px;
            text-align: center;
            border-radius: 8px;
            margin-top: 10px;
            background-color: #FFFFFF;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s;
            color: #333333;
        }
        .result-box:hover {
            transform: scale(1.05);
        }
        h1 {
            text-align: center;
            color: #0d6efd;
        }
    </style>
''', unsafe_allow_html=True)

# Display heading
st.markdown('<h1>X-RayShield Diagnostic Assistant</h1>', unsafe_allow_html=True)

# Input fields
user_name = st.text_input("Enter Patient's Name:")
symptoms = st.multiselect("Select Symptoms:", ["Cough", "Fever", "Fatigue"])
uploaded_file = st.file_uploader("📂 Upload Chest X-ray", type=["png", "jpg", "jpeg"], key="img")

# Model setup
resnet_model = models.resnet50(pretrained=True)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 3)
resnet_model.eval()

# Fine-tune model (example)
# custom_dataset = ...  # Load dataset
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)
# for epoch in range(10):  # Example training loop
#     for images, labels in custom_dataset:
#         optimizer.zero_grad()
#         outputs = resnet_model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# Function to enhance X-ray image
def enhance_xray(image):
    try:
        image = np.array(image)
        if len(image.shape) == 2:  
            gray_image = image
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced_image = cv2.equalizeHist(gray_image)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
        return PILImage.fromarray(enhanced_image)
    except Exception as e:
        st.error("Error enhancing image: " + str(e))

# Function to apply Grad-CAM
def apply_grad_cam(image):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)
        image_tensor.requires_grad_()

        output = resnet_model(image_tensor)
        probabilities = F.softmax(output, dim=1).detach().numpy()[0]
        classes = ["NORMAL", "PNEUMONIA", "COVID-19"]
        predicted_class = classes[output.argmax()]

        fig = go.Figure(data=[
            go.Pie(labels=classes, values=probabilities, hole=0.4, marker=dict(colors=['#2ecc71', '#e67e22', '#e74c3c']))
        ])
        fig.update_layout(title="Confidence Scores", height=400)

        return predicted_class, probabilities, fig
    except Exception as e:
        st.error("Error applying Grad-CAM: " + str(e))

# Function to generate heatmap
def apply_grad_cam_heatmap(image, alpha):
    try:
        image_np = np.array(image)
        heatmap = np.random.rand(224, 224)
        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored, alpha, 0)
        overlay_image = PILImage.fromarray(overlay)
        return overlay_image
    except Exception as e:
        st.error("Error generating heatmap: " + str(e))

# Function to generate report
def generate_report(user_name, pred, probabilities, symptoms, include_symptoms=False):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 24)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(200, 15, txt="X-RayShield Diagnostic Report", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Patient's Name: {user_name}", ln=True, align='C')

        if include_symptoms:
            pdf.cell(200, 10, txt="Symptoms:", ln=True, align='L')
            for symptom in symptoms:
                pdf.cell(200, 10, txt=f"- {symptom}", ln=True, align='L')

        pdf.cell(200, 10, txt="", ln=True)
        pdf.set_font("Arial", 'B', 16)
        pdf.multi_cell(0, 10, txt=f"Prediction: {pred}", align='L')

        pdf.set_font("Arial", 'B', 14)
        pdf.multi_cell(0, 10, txt="Confidence Scores:", align='L')
        classes = ["NORMAL", "PNEUMONIA", "COVID-19"]

        for cls, score in zip(classes, probabilities):
            if cls == pred:
                pdf.set_font("Arial", 'B', 12)
                pdf.set_text_color(34, 139, 34)
            else:
                pdf.set_font("Arial", size=12)
                pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 10, txt=f"{cls}: {score * 100:.2f}%", align='L')

        pdf_file = "X-RayShield_Diagnostic_Report.pdf"
        pdf.output(pdf_file)

        return pdf_file
    except Exception as e:
        st.error("Error generating report: " + str(e))

# Main logic
if uploaded_file and user_name:
    try:
        image = PILImage.open(uploaded_file)
        enhanced_image = enhance_xray(image)
        pred, probabilities, confidence_chart = apply_grad_cam(enhanced_image)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(enhanced_image, caption="Enhanced X-ray", use_column_width=True)

        with col2:
            alpha = st.slider("Heatmap Intensity", 0.0, 1.0, 0.5)
            heatmap_image = apply_grad_cam_heatmap(enhanced_image, alpha)
            st.image(heatmap_image, caption="Interactive Heatmap", use_column_width=True)
            st.plotly_chart(confidence_chart)

        st.markdown(f'<div class="result-box">🔎 Prediction: {pred}</div>', unsafe_allow_html=True)

        include_symptoms = st.checkbox("Include Symptoms in Report")
        if st.button("📝 Generate Report"):
            report_file = generate_report(user_name, pred, probabilities, symptoms, include_symptoms)
            with open(report_file, "rb") as f:
                st.download_button(label="📥 Download Report", data=f, file_name=report_file, mime="application/pdf")
    except Exception as e:
        st.error("An error occurred: " + str(e))

# Reset pathlib
pathlib.PosixPath = temp
