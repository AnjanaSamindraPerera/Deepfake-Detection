from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

import torch
from network.classifier import *
from network.transform import mesonet_data_transforms

# Imports PIL module 
from PIL import Image

#Creat the model
model = Meso4()
model.load_state_dict(torch.load("/hdd/2017CS128/Research/MesoNet-Pytorch/output/Mesonet-RESRGAN3_DROPOUT0.1/29_meso4.pkl"))
print(model)
#model = resnet18(pretrained=True).eval()
cam_extractor = SmoothGradCAMpp(model,'conv4')
# Get your input
img = read_image("/hdd/2017CS128/Research/FFc40-RESRGAN3/test/df/1_4_out.jpg")
#img = Image.open("/hdd/2017CS128/Research/FFc40-RESRGAN3/test/df/1_4_out.jpg").convert('RGB')
# Preprocess it for your chosen model
input_tensor = normalize(resize(img, (256, 256)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))
print(out)
# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)


# Visualize the raw CAM
plt.imshow(activation_map[0].numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
plt.savefig('Meso_raw_CAM.png') 

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0], mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
plt.savefig('Meso_overlay.png') 