# Number Recognition Using ANN

This application is a number recognition tool that utilizes deep learning models to classify hand-drawn digits. Built with Streamlit, it provides an interactive way to test a Convolutional Neural Network (CNN) model trained on the MNIST dataset.

## Features

- Users can draw digits (0-9) on a canvas.
- The application processes the drawing and makes a prediction using a trained CNN model.
- Compares predictions from two different models.
- Displays the resized image and confidence scores for each prediction.

## Technologies Used

- **Python**: Primary programming language.
- **Streamlit**: Web-based interactive UI framework.
- **OpenCV**: Image processing.
- **NumPy & Pandas**: Data manipulation.
- **Keras & TensorFlow**: Deep learning framework for model inference.
- **PIL (Pillow)**: Image handling.
- **streamlit-drawable-canvas**: Interactive drawing canvas.

## Installation

To run this application locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/WillOfHeaven/MinorGP.git
   cd MinorGP
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```sh
   streamlit run app.py
   ```

## Usage

1. Open the web interface.
2. Draw a digit (0-9) on the canvas.
3. The application will process and predict the number.
4. Compare results between different models.

## Model Details

- **New Model**: `new_mnist(4x4)_epoch70.h5`
- **Old Model**: `mnist.h5`
- Both models were trained on the MNIST dataset but have different architectures and training parameters.

## Retraining the Model

To improve the model performance, you can retrain it using additional data or fine-tune its parameters.

### Steps for Retraining:

1. **Prepare Data**
   - Collect additional handwritten digits.
   - Preprocess images (resize to 28x28, normalize, grayscale conversion).

2. **Load Existing Model**
   ```python
   from keras.models import load_model
   model = load_model("new_mnist(4x4)_epoch70.h5")
   ```

3. **Fine-Tune or Train from Scratch**
   - Freeze earlier layers if using transfer learning.
   - Use an augmented dataset to avoid overfitting.
   ```python
   from keras.preprocessing.image import ImageDataGenerator

   datagen = ImageDataGenerator(
       rotation_range=10,
       zoom_range=0.1,
       width_shift_range=0.1,
       height_shift_range=0.1
   )

   model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))
   ```

4. **Save the Updated Model**
   ```python
   model.save("updated_mnist_model.h5")
   ```

5. **Deploy the Updated Model**
   - Replace `new_mnist(4x4)_epoch70.h5` with `updated_mnist_model.h5` in `app.py`.
   - Restart the application.

## Contributors

- Gaurav Rawat ( Alternative Solution explorer ) 
- Karthik Sharma Dhulipati (  ML Research ) 
- Mohak Kumar Srivastava ( UI ) 
- Naman Jain ( Research ) 
- Sambuddha Chatterjee ( ML Pipeline ) 

## Feedback

We welcome your feedback and contributions! If you encounter any issues or have suggestions for improvements, feel free to submit an issue or pull request on [GitHub](https://github.com/WillOfHeaven/MinorGP).

