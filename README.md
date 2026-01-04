# kidney_Stone_Detection
# Glaucoma Eye Disease Detector

A machine learning project to detect glaucoma from eye images using TensorFlow/Keras.

## Features
- Classifies eye images as **Normal** or **Glaucoma**
- Includes pre-processing and training scripts
- Provides a pre-trained model for quick predictions

## Folder Structure
- `src/` → Python scripts (training, prediction)
- `notebooks/` → Jupyter notebooks with analysis
- `models/` → Saved models (.h5)
- `assets/` → Sample eye images
- `docs/` → Screenshots, diagrams

## How to Run
1. Install required packages:
pip install -r requirements.txt
2. Run training (optional):
python src/train_glaucoma_model.py
3. Run prediction:
python src/predict_glaucoma.py
4. Run Streamlit app:
streamlit run app.py
