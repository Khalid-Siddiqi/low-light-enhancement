from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import uuid
import os
import torch
import rawpy
import imageio
import numpy as np
from models.lsid import lsid

app = FastAPI()

CHECKPOINT_PATH = "epoch_4000_512.pth.tar"
OUTPUT_DIR = "output"

def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)
    H, W = im.shape
    out = np.stack((im[0:H:2, 0:W:2],
                    im[0:H:2, 1:W:2],
                    im[1:H:2, 1:W:2],
                    im[1:H:2, 0:W:2]), axis=2)
    return out

@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4()}.ARW"
    temp_path = os.path.join(OUTPUT_DIR, temp_filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Prepare device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lsid(inchannel=4, block_size=2).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load and preprocess raw image
    with rawpy.imread(temp_path) as raw:
        input_full = np.expand_dims(pack_raw(raw), axis=0)
    input_tensor = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        output = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output = np.clip(output, 0, 1)

    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = temp_filename.replace(".ARW", ".png")
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    imageio.imwrite(output_path, (output * 255).astype(np.uint8))

    # Cleanup temp raw file
    os.remove(temp_path)

    return FileResponse(output_path, media_type="image/png", filename=output_filename)
