import tensorflow as tf

MODEL_PATH = "kidney_stone_detector.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Model loading failed!")
    print(e)
