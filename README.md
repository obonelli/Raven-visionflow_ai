# 🧠 Pixel Watcher AI --- Raven Visionflow

**Pixel Watcher AI** is an experimental application built with
**Python + PyQt5**, focused on real-time visual analysis.\
Its main goal is to **detect visual events or anomalies on the screen**
using **Siamese neural networks** trained locally with custom datasets.

------------------------------------------------------------------------

## ⚙️ Key Features

-   🖼️ **Real-time visual monitoring** with difference mapping.\
-   🧩 **Incremental training** using positive and negative datasets.\
-   💬 **Natural language interaction** (simple intents to adjust
    sensitivity or focus).\
-   🧠 **Siamese model** powered by visual embeddings (ResNet18
    backbone).\
-   ⚡ Dynamic configuration for brightness threshold, pixel
    sensitivity, and minimum change area.\
-   💾 **Modern UI** in PyQt5 with *Monitoring*, *Configuration*, and
    *Active Training* tabs.

------------------------------------------------------------------------

## 🧰 Project Structure

    VISIONFLOW_AI/
    │
    ├── dataset/                # Positive and negative images
    │   ├── positives/
    │   └── negatives/
    │
    ├── models/                 # Trained models (not versioned)
    │
    ├── ui/                     # Graphical interface modules
    │   ├── tab_monitor.py
    │   ├── tab_config.py
    │   ├── tab_training.py
    │   ├── helpers.py
    │   └── main_window.py
    │
    └── watcher/                # Core logic and AI engine
        ├── brain/
        │   ├── model.py
        │   ├── inference.py
        │   ├── intent_processor.py
        │   └── dataset_watcher.py
        ├── capture.py
        ├── monitor_thread.py
        └── config.py

------------------------------------------------------------------------

## 🚀 How to Run Locally

1.  Install dependencies:

``` bash
pip install -r requirements.txt
```

2.  Launch the application:

``` bash
python -m watcher.main
```

3.  Select the region to monitor and fine-tune the parameters in the
    **Configuration** tab.

------------------------------------------------------------------------

## 🧩 Upcoming Improvements

-   Progressive fine-tuning with reinforcement.\
-   Integration with external notifications (Discord / Telegram).\
-   Automatic dataset export and labeling.\
-   Context-aware learning (interpreting intent).

------------------------------------------------------------------------

## 🖋️ Authors

Developed by **Oscar Bonelli** and **María Suárez**\
Under the **Raven Visionflow AI** ecosystem 🐦

> "Seeing is not enough. AI must understand what it sees."
