# fastapi_app.py
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import torch
import numpy as np
import rawpy
import imageio
import os
import uuid

from model import SeeInTheDarkModel  # assuming this is the model class name

app = FastAPI()

# Load model once at startup
model_path = "epoch_4000_512.pth.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = SeeInTheDarkModel()
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()


def preprocess_raw(file):
    with rawpy.imread(file) as raw:
        raw_image = raw.raw_image_visible.astype(np.float32)
        raw_image = np.maximum(raw_image - 512, 0) / (16383 - 512)  # Normalize
        raw_image = np.expand_dims(raw_image, axis=0)  # (1, H, W)
        raw_image = np.expand_dims(raw_image, axis=0)  # (1, 1, H, W)
        return torch.from_numpy(raw_image).to(device)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_input = f"temp_{uuid.uuid4()}.arw"
    temp_output = f"output_{uuid.uuid4()}.png"

    with open(temp_input, "wb") as f:
        f.write(await file.read())

    input_tensor = preprocess_raw(temp_input)

    with torch.no_grad():
        output = model(input_tensor)
        output = output.clamp(0, 1).cpu().squeeze().permute(1, 2, 0).numpy()
        imageio.imwrite(temp_output, (output * 255).astype(np.uint8))

    os.remove(temp_input)
    return FileResponse(temp_output, media_type="image/png", filename="enhanced.png")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
