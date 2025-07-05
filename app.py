from flask import Flask, request, render_template
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from PIL import Image, ImageOps
import numpy as np
import base64
import re
import io

app = Flask(__name__)

def create_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    
    model.add(Dense(32))
    model.add(Activation("relu"))
    
    # Output Layer
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    return model

# def load_model_safe():
#     try:
#         return load_model("digit_recognition_model.keras")
#     except:
#         print("Warning: Could not load existing model. Creating new model...")
#         model = create_model()
#         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        
#         from keras.datasets import mnist
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
#         # Normalize the data
#         x_train = x_train.astype('float32') / 255.0
#         x_test = x_test.astype('float32') / 255.0
        
#         # Reshape for CNN
#         x_train = x_train.reshape(-1, 28, 28, 1)
#         x_test = x_test.reshape(-1, 28, 28, 1)
        
#         print("Training new model...")
#         model.fit(x_train, y_train, epochs=3, validation_split=0.2, verbose=1)
        
#         # Save the new model
#         model.save("digit_recognition_model_new.keras")
#         print("New model saved as digit_recognition_model_new.keras")
        
#         return model

model = load_model("digit_recognition_model.keras")

def preprocess_digit(image):
    # Invert the image so that digits are white on black background
    inverted = ImageOps.invert(image)
    img_array = np.array(inverted).astype("float32") / 255.0
    
    # Find the bounding box of the digit
    threshold = 0.1
    binary = img_array > threshold
    coords = np.column_stack(np.where(binary))
    
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cropped = inverted.crop((x0, y0, x1 + 1, y1 + 1))
        resized = cropped.resize((20, 20), Image.LANCZOS)
    else:
        resized = inverted.resize((20, 20), Image.LANCZOS)
    
    # Center the digit in a 28x28 image
    new_image = Image.new("L", (28, 28))
    offset = ((28 - 20) // 2, (28 - 20) // 2)
    new_image.paste(resized, offset)
    
    # Convert to array and normalize to match training data
    final_array = np.array(new_image).astype("float32") / 255.0
    final_array = final_array.reshape(1, 28, 28, 1)
    return final_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        data_url = request.form.get("image")
        if data_url:
            img_str = re.search(r'base64,(.*)', data_url).group(1)
            img_bytes = base64.b64decode(img_str)
            image = Image.open(io.BytesIO(img_bytes)).convert("L")
            processed_image = preprocess_digit(image)
            prediction = model.predict(processed_image).argmax()
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)