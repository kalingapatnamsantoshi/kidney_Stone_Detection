import os

# ✅ Check if 'models' folder exists
if os.path.exists("models"):
    print("✅ 'models' folder found!")
else:
    print("❌ 'models' folder NOT found!")

# ✅ Check if kidney_stone_model.h5 file exists
if os.path.exists("models/kidney_stone_model.h5"):
    print("✅ 'kidney_stone_model.h5' file found!")
else:
    print("❌ 'kidney_stone_model.h5' file NOT found!")
