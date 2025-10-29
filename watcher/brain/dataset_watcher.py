import time
from threading import Thread
from watcher import config, utils
from .model import train_siamese_model

last_counts = {"pos": 0, "neg": 0}


def watch_dataset(model_ref):
    global last_counts
    while True:
        pos_count = len(utils.list_images(config.POS_DIR))
        neg_count = len(utils.list_images(config.NEG_DIR))
        if pos_count != last_counts["pos"] or neg_count != last_counts["neg"]:
            print("[Siamese] Nuevas imágenes detectadas. Reentrenando...")
            model_ref = train_siamese_model(config.POS_DIR, config.NEG_DIR)
            last_counts = {"pos": pos_count, "neg": neg_count}
        time.sleep(5)


def start_dataset_watcher(model_ref):
    t = Thread(target=watch_dataset, args=(model_ref,), daemon=True)
    t.start()
