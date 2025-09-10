import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from segmentation_models_pytorch import Unet

# --------------------------
# Step 1: Parse arguments
# --------------------------
parser = argparse.ArgumentParser(description="Generative Background Replacement")
parser.add_argument("--input", type=str, required=True, help="Path to input image")
parser.add_argument("--prompt", type=str, required=True, help="Text prompt for background")
parser.add_argument("--output", type=str, default="output.png", help="Path to save result")
args = parser.parse_args()

# --------------------------
# Step 2: Load image
# --------------------------
input_image = cv2.imread(args.input)
if input_image is None:
    raise FileNotFoundError(f"Image not found: {args.input}")
input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# --------------------------
# Step 3: Simple segmentation (foreground mask)
# --------------------------
# For demo: use a pretrained segmentation model (Unet with ResNet backbone)
seg_model = Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=1, activation="sigmoid")
# NOTE: For real use, load pretrained weights or replace with DeepLab/Mask2Former.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

img_tensor = transform(Image.fromarray(input_rgb)).unsqueeze(0)
mask_pred = seg_model(img_tensor)  # [1,1,H,W]
mask = (mask_pred.squeeze().detach().numpy() > 0.5).astype(np.uint8) * 255
mask = cv2.resize(mask, (input_rgb.shape[1], input_rgb.shape[0]))

# Extract foreground
foreground = cv2.bitwise_and(input_rgb, input_rgb, mask=mask)

# --------------------------
# Step 4: Load Stable Diffusion + ControlNet
# --------------------------
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

# Prepare ControlNet conditioning image (segmentation map)
cond_image = Image.fromarray(mask).convert("RGB")

# --------------------------
# Step 5: Generate new background
# --------------------------
generated = pipe(args.prompt, image=cond_image, num_inference_steps=30).images[0]
generated = np.array(generated)

# --------------------------
# Step 6: Blend foreground + new background
# --------------------------
mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) // 255
final = (foreground * mask_3c) + (generated * (1 - mask_3c))

# Save result
cv2.imwrite(args.output, cv2.cvtColor(final.astype(np.uint8), cv2.COLOR_RGB2BGR))
print(f"✅ Saved output to {args.output}")
