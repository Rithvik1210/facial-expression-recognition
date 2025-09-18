import sys, tensorflow as tf
import numpy as np
from PIL import Image

img = Image.open(sys.argv[1]).convert("L").resize((28,28))
x = np.expand_dims(np.array(img)/255.0, axis=(0,-1))
model = tf.keras.models.load_model("emotion_model.h5")
print("Predicted class:", model.predict(x).argmax())
