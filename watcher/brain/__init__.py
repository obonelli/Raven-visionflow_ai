from .model import load_siamese_model
from .dataset_watcher import start_dataset_watcher
from .inference import ai_decide

# Carga inicial del modelo y arranca el watcher
model = load_siamese_model()
start_dataset_watcher(model)
