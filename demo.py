import os
import torch
import rawpy
import imageio
import numpy as np
from models.lsid import lsid
import utils

# ======== USER CONFIGURATION ========
input_file = "test/Sony-RX1RMark2-Shotkit.ARW"  # Path to input ARW image
checkpoint_path = "checkpoint.pth.tar"       # Path to model checkpoint
output_dir = "output/"                            # Path to output directory
output_name = os.path.splitext(os.path.basename(input_file))[0] + ".png"
output_path = os.path.join(output_dir, output_name)

# ======== IMAGE PREPROCESSING ========
def pack_raw(raw):
    # Pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # Normalize
    img_shape = im.shape
    H, W = img_shape[0], img_shape[1]
    out = np.stack((im[0:H:2, 0:W:2],
                    im[0:H:2, 1:W:2],
                    im[1:H:2, 1:W:2],
                    im[1:H:2, 0:W:2]), axis=2)
    return out

# ======== MAIN INFERENCE FUNCTION ========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load raw image
    with rawpy.imread(input_file) as raw:
        input_full = np.expand_dims(pack_raw(raw), axis=0)

    input_full = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device)

    # Load model
    model = lsid(inchannel=4, block_size=2).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)


    # Load checkpoint directly (weights are not wrapped in a named dictionary)
    model.load_state_dict(checkpoint["model_state_dict"])


    model.eval()

    # Run inference
    with torch.no_grad():
        output = model(input_full)
        output = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output = np.clip(output, 0, 1)

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    imageio.imwrite(output_path, (output * 255).astype(np.uint8))
    print(f"Saved output to: {output_path}")

if __name__ == "__main__":
    main()
