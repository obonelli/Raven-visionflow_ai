import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
from watcher import config, utils

# ===== Configuración general =====
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/siamese_model.pt"

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])


# ===== Modelo base =====
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 64))

    def forward_once(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


# ===== Funciones de utilidad =====
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img if img is not None else None


def embed_image(model, img_bgr):
    model.eval()
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.forward_once(tensor)
    feat = nn.functional.normalize(feat, p=2, dim=1)
    return feat.cpu().numpy()[0]


# ===== Entrenamiento del modelo =====
def train_siamese_model(pos_dir, neg_dir, epochs=5):
    pos_files = utils.list_images(pos_dir)
    neg_files = utils.list_images(neg_dir)

    if len(pos_files) < 2 or len(neg_files) < 2:
        print("[Siamese] Dataset insuficiente, creando modelo vacío temporal.")
        return SiameseNetwork().to(device)

    model = SiameseNetwork().to(device)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    for epoch in range(epochs):
        total_loss = 0
        for pos_path in pos_files:
            pos_img = load_image(pos_path)
            neg_img = load_image(np.random.choice(neg_files))
            if pos_img is None or neg_img is None:
                continue

            pos_tensor = (
                transform(Image.fromarray(cv2.cvtColor(pos_img, cv2.COLOR_BGR2RGB)))
                .unsqueeze(0)
                .to(device)
            )
            neg_tensor = (
                transform(Image.fromarray(cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)))
                .unsqueeze(0)
                .to(device)
            )

            out1, out2p = model(pos_tensor, pos_tensor)
            out1, out2n = model(pos_tensor, neg_tensor)

            loss_pos = criterion(out1, out2p, torch.tensor([1.0], device=device))
            loss_neg = criterion(out1, out2n, torch.tensor([-1.0], device=device))
            loss = (loss_pos + loss_neg) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Siamese] Época {epoch+1}/{epochs} | Pérdida: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("[Siamese] Modelo guardado:", MODEL_PATH)
    return model


# ===== Carga del modelo entrenado =====
def load_siamese_model():
    if os.path.exists(MODEL_PATH):
        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[Siamese] Modelo cargado correctamente.")
    else:
        model = train_siamese_model(config.POS_DIR, config.NEG_DIR)
    return model


# ===== Instancia global del modelo =====
siamese = load_siamese_model()
