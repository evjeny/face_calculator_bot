import os
import shutil
import ast
from io import BytesIO
from threading import Timer
from time import time

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from telegram.update import Update
import logging

from net.net import CNNVAE


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

storage_path = os.path.join(os.getcwd(), "storage")
files_queue = []

image_size = 64
latent_size = 256


def singleton(class_):
    instances = {}

    def wrapper(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return wrapper


model = singleton(CNNVAE)("weights/cnnvae_64px.pt")


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        logger.info("init timer with function {}".format(self.function.__name__))

        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        logger.info("repeating function {}".format(self.function.__name__))
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def clean_all_files(exclude=()):
    logger.info("cleaning all files")
    for file in os.listdir(storage_path):
        if file not in exclude:
            shutil.rmtree(os.path.join(storage_path, file))


def add_file_to_queue(user_id, variable_name):
    logger.info("adding file {} of user {} to queue".format(variable_name, user_id))
    path = os.path.join(storage_path, str(user_id), variable_name)
    current_time = int(time())
    files_queue.append((current_time, path))


def remove_from_queue(lifetime):
    global files_queue

    files_count = len(files_queue)
    current_time = int(time())
    i = 0
    while i < len(files_queue):
        add_time, file_path = files_queue[i]
        if current_time - add_time >= lifetime:
            os.remove(file_path) # remove too old files from storage
            logger.info("removed too old file {} from memory".format(file_path))
        else:
            break
        i += 1
    files_queue = files_queue[i:] # remove information about old files from storage
    logger.info("removed {} files from queue".format(files_count - len(files_queue)))


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

    tensor_result = predict_variable(var_image)

    torch.save(tensor_result, var_path)
    logger.info("saved variable {} of user {}".format(var_name, user_id))


def predict_variable(pil_image):
    transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    tensor_image = transformation(pil_image).view(1, 3, image_size, image_size)
    return model.reparam(*model.encoder(tensor_image))


def read_variable(user_id, variable_name):
    user_path = os.path.join(storage_path, str(user_id))
    variable = torch.load(os.path.join(user_path, variable_name))
    logger.info("read variable {} of user {}".format(variable_name, user_id))

    return variable


def get_image_from_variable(variable):
    back_transformation = transforms.ToPILImage(mode="RGB")

    tensor_image = model.decoder(variable)
    image = back_transformation(tensor_image[0])

    return image


def get_variable_names(expression):
    names = [
        node.id for node in ast.walk(ast.parse(expression))
        if isinstance(node, ast.Name)
    ]
    logger.info("got variable names: {}".format(names))

    return names


def read_variables(user_id, variable_names):
    variables = [(name, read_variable(user_id, name)) for name in variable_names]
    variables = dict(variables)
    logger.info("read variables {} of user {}".format(variable_names, user_id))

    return variables


def convert_image_to_io(image, name="_.jpeg"):
    bio = BytesIO()
    bio.name = name
    image.save(bio, "JPEG")
    bio.seek(0)

    return bio


def handle_message_edit(handler_function):
    def wrapper(*args, **kwargs):
        for arg in args + tuple(kwargs.values()):
            if isinstance(arg, Update) and not arg.message:
                logger.info("handled message edit in function {}".format(handler_function.__name__))
                return
        handler_function(*args, **kwargs)
    return wrapper


def steps_between_tensors(var_a: torch.Tensor, var_b: torch.Tensor, steps=10) -> torch.Tensor:
    size: int = np.prod(var_a.size())
    var_a = var_a.view(size).detach().numpy()
    var_b = var_b.view(size).detach().numpy()
    result = torch.zeros(size, steps)

    for i in range(size):
        result[i] = torch.linspace(var_a[i], var_b[i], steps)
    return result.t()


