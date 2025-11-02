import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from watcher import config

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/combo_model.pt"

# ======================================================
# 🔹 Transformaciones (con aumento de datos robusto)
# ======================================================
transform = transforms.Compose(
    [
        transforms.Resize((96, 96)),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05
                ),
            ],
            p=0.9,
        ),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(size=(96, 96), scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ======================================================
# 🔹 Definición del modelo
# ======================================================
class ComboNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Linear(base.fc.in_features, 3)
        self.model = base

    def forward(self, x):
        return self.model(x)


# ======================================================
# 🔹 Entrenamiento del modelo
# ======================================================
def train_combo_model(epochs=18):
    dataset = datasets.ImageFolder(config.COMBO_DIR, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = ComboNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0004, weight_decay=1e-5)

    print(
        f"[ComboNet] Entrenando con {len(dataset)} imágenes (con aumento dinámico)..."
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(
            f"[ComboNet] Época {epoch+1}/{epochs} - pérdida: {total_loss:.4f} - acc: {acc:.2f}%"
        )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("[ComboNet] Modelo guardado:", MODEL_PATH)

    evaluate_combo_model(model, loader)
    return model


# ======================================================
# 🔹 Evaluación por clase
# ======================================================
def evaluate_combo_model(model, loader):
    model.eval()
    correct = {0: 0, 1: 0, 2: 0}
    total = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = torch.argmax(out, dim=1)
            for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                total[label] += 1
                if label == pred:
                    correct[label] += 1

    print("\n=== Precisión por clase ===")
    for cls in [0, 1, 2]:
        if total[cls] > 0:
            acc = 100 * correct[cls] / total[cls]
            print(f"Clase {cls}: {acc:.2f}% ({correct[cls]}/{total[cls]})")
        else:
            print(f"Clase {cls}: sin muestras")


# ======================================================
# 🔹 Cargar o entrenar modelo
# ======================================================
def load_combo_model():
    model = ComboNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[ComboNet] Modelo de combo cargado.")
    else:
        print("[ComboNet] Entrenando desde cero...")
        model = train_combo_model()
    model.eval()
    return model


# ======================================================
# 🔹 Ejecución directa
# ======================================================
if __name__ == "__main__":
    print(
        "[ComboNet] Iniciando entrenamiento manual del modelo de combos (con aumento)..."
    )
    train_combo_model(epochs=18)
