# 🩺 Kidney Stone Detection System using Machine Learning
# 📌 Project Overview

Kidney stones are a common and painful medical condition that often requires early detection for effective treatment.
This project presents a Kidney Stone Detection System built using Machine Learning and Deep Learning techniques to classify kidney images as Stone or Normal.
The model analyzes medical images and predicts whether a kidney stone is present, helping in early diagnosis and decision support.

# 🎯 Objectives

Detect kidney stones from medical images
Reduce manual diagnosis effort
Provide fast and reliable predictions
Build a beginner-friendly yet practical ML project

# 🛠️ Technologies Used

Python(version 3.13.1)
TensorFlow / Keras
NumPy
OpenCV
Matplotlib
Scikit-learn
Windsurf

🗂️ Project Structure
Kidney-Stone-Detector/
│
├── dataset/
│   ├── Stone/
│   └── Normal/
│
├── model/
│   └── kidney_stone_model.h5
│
├── train_model.py
├── test_model.py
├── requirements.txt
└── README.md

#📊 Dataset Information

The dataset consists of kidney medical images

Two classes:
Stone – images containing kidney stones
Normal – images without kidney stones
Dataset is not included in this repository for experimentation and learning purposes

⚠️ This dataset is intended for educational and research use only.

#⚙️ How It Works

Images are loaded and resized
Preprocessing is applied (normalization, labeling)
A Convolutional Neural Network (CNN) is trained

#The trained model predicts:
Stone
Normal

🚀 Installation & Setup
Step 1: Clone the Repository
git clone https://github.com/your-username/kidney-stone-detector.git
cd kidney-stone-detector

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Train the Model
python train_model.py

Step 4: Test the Model
python test_model.py

Step 5: Run the model
streamlit app.py

