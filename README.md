# AgriSense — AI Smart Farming Assistant

**SDG 2: Zero Hunger** | Academic full-stack AI project

This Flask website accepts a plant-leaf image, sends it to a trained TensorFlow image classifier, displays the predicted class and confidence, gives general educational care guidance, and stores recent predictions in SQLite.

## 1. Requirements

- Windows 10/11, Python 3.10–3.12 (Python 3.11 is a good choice)
- VS Code
- Internet for installing Python packages and downloading the dataset

Check Python:
```bash
python --version
```

## 2. Setup on Windows

Open this project folder in VS Code, then choose **Terminal → New Terminal**.

Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

Install packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add a plant disease dataset and train the AI

The web app is connected to a real TensorFlow model. The model files are not included in this starter ZIP because they must be trained from a dataset.

Use a legally obtained PlantVillage dataset (or another suitable, licensed leaf-image dataset). Arrange it like this:

```text
AI-Smart-Farming/
  data/
    PlantVillage/
      Tomato___healthy/
        image1.jpg
        image2.jpg
      Tomato___Early_blight/
        image1.jpg
      Tomato___Late_blight/
        image1.jpg
      Potato___healthy/
        image1.jpg
      ...
```

Each class must have its own folder, and that folder must contain images. You can include all classes in the dataset; the script reads folder names automatically.

Run training:
```bash
python train_model.py
```

The script uses transfer learning with MobileNetV2 and saves:
- `model/plant_disease_model.keras`
- `model/class_names.json`

Training can take time and uses memory. If your laptop is slow or runs out of memory, train in Google Colab and download both generated model files into this project's `model/` folder.

## 4. Run the website

```bash
python app.py
```

Open the local address printed in the terminal, usually:
`http://127.0.0.1:5000`

Upload a clear JPG, PNG or WEBP leaf image (maximum 8 MB).

## 5. How it works

1. HTML/CSS/JavaScript provide the upload interface and display results.
2. Flask receives the image at `/api/predict`.
3. Pillow resizes the image; TensorFlow loads the trained `.keras` model and predicts a class.
4. Flask returns the class, confidence and general care suggestions as JSON.
5. SQLite stores recent prediction history.

## 6. Important notes

- Before training, the website will show a clear message that the AI model is not ready. This is intentional; it does not pretend to detect diseases without a trained model.
- Model confidence is not a guarantee that a diagnosis is correct. PlantVillage-style datasets may contain clean, lab-like images and may not generalize to field photos.
- Treatment guidance is educational, not a pesticide prescription. Confirm diagnosis and treatment with a local agricultural expert and follow local regulations and product labels.
- This is an academic prototype, not a certified agricultural diagnostic service.
