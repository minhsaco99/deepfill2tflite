import torch
from PIL import Image
import sys
sys.path.append('../deepfillv2-pytorch')
from model.networks import Generator
import torchvision.transforms as T

onnx_model_path = 'gen.onnx'
model_path = '../states_pt_places2.pth'
image_path = '../deepfillv2-pytorch/examples/inpaint/case1.png'
mask_path = '../deepfillv2-pytorch/examples/inpaint/case1_mask.png'
device = 'cpu'

generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)

generator_state_dict = torch.load(model_path, map_location='cpu')['G']
generator.load_state_dict(generator_state_dict, strict=True)

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
                dim=1)  # concatenate channels

dynamic_axes = {'input':{2:'img_size', 3:'img_size'},
                'mask':{2:'img_size', 3:'img_size'},
                'x_stage1':{2:'img_size', 3:'img_size'},
                'x_stage2':{2:'img_size', 3:'img_size'}}

torch.onnx.export(
    generator,
    (x, mask), 
    onnx_model_path,
    verbose=False,
    export_params=True,
    do_constant_folding=True,
    input_names=['input', 'mask'],
    output_names=['x_stage1', 'x_stage2'],
    opset_version=11,
    dynamic_axes=dynamic_axes
)