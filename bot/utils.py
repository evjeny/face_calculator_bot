from threading import Timer
import os

from net.net import CNNVAE


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
    user_path = os.path.join("storage", str(user_id))
    os.makedirs(user_path, exist_ok=True)

    var_file.download(os.path.join(user_path, "temp"))

    model = singleton(CNNVAE)()
    

