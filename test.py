from PIL import Image
from torchvision import transforms

from net.net import CNNVAEBig


model = CNNVAEBig("weights/cnnvae_aug_256px.pt")

img = Image.open("6.jpg")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
prediction = model(transform(img).view(1, 3, 256, 256))

result = transforms.ToPILImage(mode="RGB")(prediction[0])
result.save("6_out.jpg")

