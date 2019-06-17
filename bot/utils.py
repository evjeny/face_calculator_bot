from threading import Timer
import os
import ast
from io import BytesIO

import torch
import torchvision.transforms as transforms
from PIL import Image

from net.net import CNNVAE


storage_path = "storage"
image_size = 64
latent_size = 256


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def singleton(class_):
    instances = {}

    def wrapper(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return wrapper


def clean_files(interval=30*60):
    pass


def save_variable(user_id, var_name, var_file):
    user_path = os.path.join(storage_path, str(user_id))
    temp_path = os.path.join(user_path, "temp")
    os.makedirs(user_path, exist_ok=True)

    var_file.download(temp_path)
    image = Image.open(temp_path)

    save_variable_pil(user_id, var_name, image)


def save_variable_pil(user_id, var_name, var_image):
    user_path = os.path.join(storage_path, str(user_id))
    var_path = os.path.join(user_path, var_name)
    os.makedirs(user_path, exist_ok=True)

    model = singleton(CNNVAE)("weights/cnnvae_64px.pt")

    transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    tensor_image = transformation(var_image).view(1, 3, image_size, image_size)
    tensor_result = model.reparam(*model.encoder(tensor_image))

    torch.save(tensor_result, var_path)


def read_variable(user_id, variable_name):
    user_path = os.path.join(storage_path, str(user_id))
    return torch.load(os.path.join(user_path, variable_name))


def get_image_from_variable(variable):
    model = singleton(CNNVAE)("weights/cnnvae_64px.pt")
    back_transformation = transforms.ToPILImage(mode="RGB")

    tensor_image = model.decoder(variable)
    image = back_transformation(tensor_image[0])

    return image


def get_variable_names(expression):
    names = [
        node.id for node in ast.walk(ast.parse(expression))
        if isinstance(node, ast.Name)
    ]
    return names


def read_variables(user_id, variable_names):
    variables = [(name, read_variable(user_id, name)) for name in variable_names]
    variables = dict(variables)
    return variables


def convert_image_to_io(image, name="_.jpeg"):
    bio = BytesIO()
    bio.name = name
    image.save(bio, "JPEG")
    bio.seek(0)

    return bio

