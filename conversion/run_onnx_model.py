import onnxruntime as ort

import torch
from PIL import Image
import sys
sys.path.append('../deepfillv2-pytorch')
import torchvision.transforms as T
import numpy as np
import time

onnx_model_path = 'gen.onnx'
image_path = '../deepfillv2-pytorch/examples/inpaint/case2.png'
mask_path = '../deepfillv2-pytorch/examples/inpaint/case2_mask.png'
device = 'cpu'


image = Image.open(image_path)
mask = Image.open(mask_path)

# prepare input
image = T.ToTensor()(image)
mask = T.ToTensor()(mask)

_, h, w = image.shape
grid = 8

image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

print(f"Shape of image: {image.shape}")

image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
mask = (mask > 0.5).to(dtype=torch.float32,
                        device=device)  # 1.: masked 0.: unmasked

image_masked = image * (1.-mask)  # mask image

ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
x = torch.cat([image_masked, ones_x, ones_x*mask],
                dim=1).numpy()  # concatenate channels
mask = mask.numpy()

sess = ort.InferenceSession("gen.onnx")
input_name = sess.get_inputs()
label_name = sess.get_outputs()
for i in range(5):
    start = time.time()
    x_stage1, x_stage2 = sess.run([label_name[0].name, label_name[1].name], {
                        input_name[0].name:x,
                        input_name[1].name:mask,
                        })
    print(time.time() - start)
image_inpainted = image * (1.-mask) + x_stage2 * mask
# save inpainted image
img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
img_out = img_out.to(device='cpu', dtype=torch.uint8).numpy()
img_out = Image.fromarray(img_out)
img_out.save('out.jpg')