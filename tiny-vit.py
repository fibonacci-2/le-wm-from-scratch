from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
import pandas as pd
import h5py

with h5py.File("data/tworoom.h5", "r") as f:
    # Load the dataset into a pandas DataFrame
    df = pd.DataFrame({
        'pixels': list(f['pixels'][:64]),
        'actions': list(f['action'][:64])
    })

model_name = "yujiepan/clip-vit-tiny-random-patch14-336"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(model_name)

# encode a smaple image by the model
model_input = processor(images=df.iloc[0]['image'], return_tensors="pt")

print(model)