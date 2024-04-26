# Phishing Email Detection

This project demonstrates a simple phishing email detection system using a K-Nearest Neighbors (KNN) classifier trained on a dataset of email subjects and bodies.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/phishing-email-detection.git
    ```

2. Navigate to the project directory:

    ```bash
    cd phishing-email-detection
    ```

3. Install the required dependencies:

    ```bash
    pip install -r 
    ```

## Usage

1. Ensure you have a dataset of emails in CSV format. The dataset should contain two columns: `subject` and `body`, and a column `label` indicating whether each email is legitimate or phishing.

2. Replace the `Ling.csv` file in the project directory with your dataset.

3. Run the training script to train the model and save it:

    ```bash
    python train_model.py
    ```

4. Once the model is trained, you can run the Streamlit app to interact with the trained model:

    ```bash
    streamlit run app.py
    ```

5. The Streamlit app will open in your default web browser. You can input email text to the provided text area, and the app will predict whether it is a phishing email or not.

## Files

- `train_model.py`: Script to preprocess the data, train the KNN model, and save it.
- `app.py`: Streamlit app for interacting with the trained model.
- `Ling.csv`: Sample dataset of emails (replace with your own dataset).
- `requirements.txt`: List of Python dependencies for the project.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- Streamlit

---
