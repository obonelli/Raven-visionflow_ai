import numpy as np
from .model import embed_image, load_image
from watcher import config, utils


def visual_predict(frame, model):
    """Compara la imagen actual con las carpetas de positivos y negativos."""
    pos_imgs = [load_image(p) for p in utils.list_images(config.POS_DIR)]
    neg_imgs = [load_image(p) for p in utils.list_images(config.NEG_DIR)]
    if not pos_imgs or not neg_imgs:
        return 0

    feat_frame = embed_image(model, frame)
    sims_pos, sims_neg = [], []

    for img in pos_imgs:
        if img is not None:
            sims_pos.append(np.dot(feat_frame, embed_image(model, img)))
    for img in neg_imgs:
        if img is not None:
            sims_neg.append(np.dot(feat_frame, embed_image(model, img)))

    sim_pos = np.mean(sims_pos)
    sim_neg = np.mean(sims_neg)
    print(f"[Siamese] sim_pos={sim_pos:.3f} sim_neg={sim_neg:.3f}")
    return 1 if sim_pos > sim_neg else 0


def ai_decide(frame, text):
    """Evalúa la decisión visual y textual combinada."""
    from . import model as brain_model  # import lazy para acceder al modelo cargado

    # usa directamente brain_model.SiameseNetwork, si ya está instanciado en model.py
    if hasattr(brain_model, "siamese"):
        model_ref = brain_model.siamese
    elif hasattr(brain_model, "model"):
        model_ref = brain_model.model
    else:
        # última opción: el módulo entero
        model_ref = brain_model

    visual_pred = visual_predict(frame, model_ref)
    visual_decision = visual_pred == 1

    text_decision = False
    if text and len(text) > 3:
        keywords = ["hola", "hi", "hey", "ok", "new", "msg", "mensaje", "sí", "no"]
        text_decision = any(k.lower() in text.lower() for k in keywords)

    print(f"[Debug] Visual={visual_decision} | Text={text_decision}")
    return visual_decision or text_decision
