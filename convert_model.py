from tensorflow.keras.models import load_model

# load the keras format model
model = load_model("models/final_cnn_3class_model.keras", compile=False)

# save in H5 format
model.save("models/final_cnn_3class_model.h5")

print("Model successfully converted to H5 format")