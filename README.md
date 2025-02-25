# Pneumonia Detection using FastAI ResNet50 Model 

## Overview

This project uses FastAI and ResNet50 to classify chest X-ray images into three categories:
1. Normal
2. Bacterial Pneumonia
3. Viral Pneumonia

A Gradio web interface is provided for easy image classification. Upload an X-ray image, and the model will predict the disease category and confidence scores.

## Technologies Used
- Python
- FastAI (for training and inference)
- PyTorch (deep learning framework)
- Gradio (to create a user-friendly web UI)

## Environment Setup
1. Install Dependencies
```
pip install fastai gradio kaggle numpy matplotlib pillow
```
2. Setup Kaggle API
By downloading your Kaggle API key (kaggle.json):
```
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
3. Download Dataset
```
from kaggle.api.kaggle_api_extended import KaggleApi
import os

dataset_name = "paultimothymooney/chest-xray-pneumonia"
download_folder = "dataset"

api = KaggleApi()
api.authenticate()
os.makedirs(download_folder, exist_ok=True)
api.dataset_download_files(dataset_name, path=download_folder, unzip=True)
```

## Running the Code
1. Train the Model
```
from fastai.vision.all import *

path = Path("/content/dataset/chest_xray/train")
lung_types = ['NORMAL', 'VIRUS', 'BACTERIA']

lungs = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128)
)

dls = lungs.dataloaders(path)
learn = vision_learner(dls, resnet50, metrics=accuracy)
learn.fine_tune(25)
learn.export("export.pkl")
```

2. Run the Gradio Interface
After training the model, launch the Gradio web app for predictions:
```
import gradio as gr
from fastai.vision.all import *
from PIL import Image

learn_inf = load_learner("export.pkl")

def predict_image(img):
    img = PILImage.create(img)
    pred, _, probs = learn_inf.predict(img)
    probs_formatted = {learn_inf.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}
    return f"Prediction: {pred}", probs_formatted

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Predicted Label"), gr.JSON(label="Probabilities")],
    title="Pneumonia Classification",
    description="Upload an X-ray image to classify",
)
iface.launch()
```

## Additional Information
- The model is trained on a ResNet50 backbone for image classification.
- Training is performed for 25 epochs to ensure high accuracy.

## References
- Dataset: [Chest X-ray Pneumonia on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download)
- FastAI Documentation: [FastAI](https://docs.fast.ai/)
