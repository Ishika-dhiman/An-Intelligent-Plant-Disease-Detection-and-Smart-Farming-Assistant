
"""
Train a plant-leaf image classifier using a folder-based dataset.

Expected structure:
data/PlantVillage/
    Pepper__bell___Bacterial_spot/
    Pepper__bell___healthy/
    Potato___Early_blight/
    Potato___healthy/
    Tomato_Bacterial_spot/
    Tomato_healthy/
    ...
"""

import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

# --------------------------------------------------
# CPU USAGE SETTINGS
# --------------------------------------------------

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "PlantVillage"
MODEL_DIR = ROOT / "model"

MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# TRAINING SETTINGS
# --------------------------------------------------

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 8
SEED = 123
EPOCHS = 3

# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------

if not DATA_DIR.exists():
    raise SystemExit(
        "Dataset folder not found. "
        "Please create data/PlantVillage/"
    )

# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

if len(class_names) < 2:
    raise SystemExit(
        "Error: At least 2 disease classes are required."
    )

print("Total disease classes:", len(class_names))
print("Disease names:", class_names)

# Save class names
(MODEL_DIR / "class_names.json").write_text(
    json.dumps(class_names, indent=2),
    encoding="utf-8"
)

# --------------------------------------------------
# DATA PERFORMANCE SETTINGS
# --------------------------------------------------

train_ds = train_ds.prefetch(1)
val_ds = val_ds.prefetch(1)

# --------------------------------------------------
# LOAD MOBILENETV2 MODEL
# --------------------------------------------------

base = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base.trainable = False

# --------------------------------------------------
# BUILD CLASSIFICATION MODEL
# --------------------------------------------------

inputs = tf.keras.Input(shape=(160, 160, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    inputs
)

x = base(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

# --------------------------------------------------
# COMPILE MODEL
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nStarting lightweight model training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------

loss, accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print(
    f"\nValidation accuracy: {accuracy * 100:.2f}%"
)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model.save(
    MODEL_DIR / "plant_disease_model.keras"
)

print("\nModel saved successfully!")
print("Saved model to model/plant_disease_model.keras")
print("Saved class labels to model/class_names.json")
