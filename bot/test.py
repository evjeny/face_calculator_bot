from PIL import Image

import torchvision.transforms as transforms

from net.net import CNNVAE


model = CNNVAE("weights/cnnvae_64px.pt")
img = Image.open("6.jpg")

transformation = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

back_transformation = transforms.ToPILImage(mode="RGB")

tensor_image = transformation(img).view(1, 3, 64, 64)
prediction = back_transformation(model(tensor_image)[0])

prediction.save("6_out.jpg")

