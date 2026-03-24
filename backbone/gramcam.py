# Here is the code ：

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def main():
    model = models.resnet50(pretrained=True)
    weights_path = r"F:\PM_success\MobileViT\resnet\LOSO_2.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    target_layers = [model.layer4[-1]]
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Prepare image
    img_path = r"F:\PM_success\MobileViT\moblie_repvit\img86.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # Grad CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    # targets = [ClassifierOutputTarget(281)]     # cat
    targets =  None     # dog

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.,
                                      grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
