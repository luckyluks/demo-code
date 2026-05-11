import os
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import v2 as transforms
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet18_Weights,
    EfficientNet_B0_Weights,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output


from pathlib import Path
import shutil
import urllib.request
import zipfile


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Dataset configuration
dataset_factor_default = (
    1  # when you change this, remember to change the same in the URI in the Resetdag
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

### begin old section; Ordnerpfad zum Dataset

# Widgets for data source selection (used in the next data-loading cell)
data_root_w = widgets.Text(
    value="dataset/", description="Data root:", layout=widgets.Layout(width="500px")
)
initial_dir_w = widgets.Text(
    value="dataset/", description="Image dir:", layout=widgets.Layout(width="500px")
)
ann_json_dir_w = widgets.Text(
    value="", description="Ann JSON:", layout=widgets.Layout(width="600px")
)
ann_img_dir_w = widgets.Text(
    value="", description="Ann images:", layout=widgets.Layout(width="600px")
)
use_ann_w = widgets.Checkbox(value=False, description="Use human annotations")

# data_root_w = "dataset/"
# initial_dir_w = "dataset/"
# ann_json_dir_w = ""
# ann_img_dir_w = ""
# use_ann_w = False

### end old section; Ordnerpfad zum Dataset

### begin old section; Funktionen um die Bilder zuzuscheiden und die Labels zu zählen

IMG_SIZE = 224
CROP_SIZE = 672


def center_crop_resize(pil_image, crop_size=CROP_SIZE, out_size=IMG_SIZE):
    pil_image = pil_image.convert("RGB")
    w, h = pil_image.size
    cs = min(crop_size, w, h)
    left = (w - cs) // 2
    top = (h - cs) // 2
    cropped = pil_image.crop((left, top, left + cs, top + cs))
    return cropped.resize((out_size, out_size), Image.Resampling.BILINEAR)


def sample_count_dict(labels):
    counts = defaultdict(int)
    for y in sorted(labels):
        counts[y] += 1
    return dict(counts)


def plot_split_class_distributions(train_counts, val_counts, test_counts):
    """Plot class-count bar charts for train, validation, and test splits.

    Args:
        train_counts: Dict like {"class_name": count} for the train split.
        val_counts: Dict like {"class_name": count} for the validation split.
        test_counts: Dict like {"class_name": count} for the test split.
    """
    split_payload = [
        ("Classes in train set", train_counts),
        ("Classes in val set", val_counts),
        ("Class distribution in test set", test_counts),
    ]

    all_classes = set()
    for _, counts in split_payload:
        all_classes.update(counts.keys())

    class_names = sorted(all_classes)
    if not class_names:
        print("No class counts provided.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, (title, counts), color in zip(axes, split_payload, colors):
        values = [counts.get(class_name, 0) for class_name in class_names]
        bars = ax.bar(class_names, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()


def display_training_status(
    epoch,
    epochs,
    selected_model_name,
    selected_training_mode,
    selected_optimizer_name,
    lr,
    current_lr,
    tr_loss,
    tr_acc,
    va_loss,
    va_acc,
    conf_weighted_acc,
    hist,
):
    """Print training info and plot live metrics for an epoch.

    This centralizes the display logic so notebooks can call it from their
    training loops and avoid duplicate UI/Output code.
    """
    clear_output(wait=True)

    print("Training Setup: " + "-" * 50)
    print(
        f"Model Details: model={selected_model_name}, mode={selected_training_mode}, optimizer={selected_optimizer_name}"
    )
    if selected_optimizer_name == "sgd":
        print(f"SGD momentum: {sgd_momentum_w.value:.2f}")
    print(f"Training Parameters: lr={lr}, epochs={epochs}")
    print(
        f"Data Details: factor={dataset_factor_w.value}, Val_split={val_split_w.value}, test_split={test_split_w.value}, batch_size={batch_w.value}, hflip={hflip_w.value}, vflip={vflip_w.value}"
    )
    try:
        print(
            f"Dataset Details: Train samples={len(train_loader.dataset)}, Val samples={len(val_loader.dataset)}, Test samples={len(test_loader.dataset)}"
        )
    except Exception:
        pass

    print("Live Metrics: " + "-" * 50)
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train loss: {tr_loss:.4f} | Train acc: {tr_acc:.2f}%")
    print(
        f"Val   loss: {va_loss:.4f} | Val   acc: {va_acc:.2f}% | Val Conf-Weighted Acc: {conf_weighted_acc:.2f}%"
    )
    print(f"Current LR: {current_lr:.6f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["train_loss"], label="train_loss")
    ax[0].plot(hist["val_loss"], label="val_loss")
    ax[0].legend()
    ax[0].set_title("Loss")

    ax[1].plot(hist["train_acc"], label="train_acc")
    ax[1].plot(hist["val_acc"], label="val_acc")
    ax[1].legend()
    ax[1].set_title("Accuracy")
    plt.show()


def get_class_names():
    return list(globals().get("class_names", sorted(class_dict.keys())))


def preprocess_pil_for_inference(pil_image):
    im = pil_image.convert("RGB")
    im = center_crop_resize(im)
    _, eval_tfm_local = get_transforms(hflip=0.0, vflip=0.0)
    x = eval_tfm_local(im).unsqueeze(0).to(device)
    return im, x


def predict_pil_image(pil_image, model):
    model.eval()
    im, x = preprocess_pil_for_inference(pil_image)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx_to_class = {v: k for k, v in class_dict.items()}
    pred_idx = int(np.argmax(probs))
    pred_class = idx_to_class[pred_idx]
    return im, pred_class, probs


def select_one_sample_per_class(class_names=None, dataset_dir="dataset"):
    class_names = list(class_names or get_class_names())
    selected_samples = []
    class_to_indices = {class_name: [] for class_name in class_names}

    if (
        hasattr(__import__(__name__), "x_test")
        and hasattr(__import__(__name__), "y_test")
        and len(x_test) > 0
    ):
        for idx, label in enumerate(y_test):
            if label in class_to_indices:
                class_to_indices[label].append(idx)

        for class_name in class_names:
            candidates = class_to_indices.get(class_name, [])
            if candidates:
                chosen_idx = int(np.random.choice(candidates))
                selected_samples.append(
                    {
                        "class": class_name,
                        "pil": x_test[chosen_idx],
                        "source": "test set",
                        "index": chosen_idx,
                    }
                )

    missing_classes = [
        class_name
        for class_name in class_names
        if class_name not in {sample["class"] for sample in selected_samples}
    ]

    if missing_classes:
        image_files = [
            path
            for path in Path(dataset_dir).rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

        files_by_class = {}
        for path in image_files:
            try:
                label = extract_label_from_filename(path.name)
                files_by_class.setdefault(label, []).append(path)
            except Exception:
                continue

        for class_name in missing_classes:
            class_files = files_by_class.get(class_name, [])
            if class_files:
                chosen_path = class_files[int(np.random.randint(len(class_files)))]
                selected_samples.append(
                    {
                        "class": class_name,
                        "pil": Image.open(chosen_path).convert("RGB"),
                        "source": f"dataset file: {chosen_path}",
                        "index": None,
                    }
                )

    class_counts = {
        class_name: len(indices) for class_name, indices in class_to_indices.items()
    }
    test_sample_count = len(y_test) if "y_test" in globals() else 0
    print(f"Available classes: {class_counts} from {test_sample_count} testset samples")

    return selected_samples


def plot_prediction_samples(
    selected_samples, model, title_fontsize=18, details_fontsize=15
):
    if not selected_samples:
        raise ValueError("No samples found for prediction.")

    num_images = len(selected_samples)
    cols = min(4, num_images)
    rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.array(axes).reshape(rows, cols)

    idx_to_class = {v: k for k, v in class_dict.items()}

    for idx, sample in enumerate(selected_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        im, pred_class, probs = predict_pil_image(sample["pil"], model)
        truth = sample["class"]

        ax.imshow(im)
        ax.set_title(f"Pred: {pred_class}\nTruth: {truth}", fontsize=title_fontsize)
        ax.axis("off")

        prob_lines = [
            f"{idx_to_class[class_idx]}: {prob * 100:.1f}%"
            for class_idx, prob in sorted(
                enumerate(probs), key=lambda item: item[1], reverse=True
            )
        ]
        details_text = "\n".join(
            [
                f"Source: {sample['source']} ({sample['index']})",
                f"Truth: {truth}",
                f"Pred: {pred_class}",
                "Probabilities:",
                *prob_lines,
            ]
        )

        ax.text(
            0.5,
            -0.05,
            details_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=details_fontsize,
        )

    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.65)
    plt.show()


def display_one_prediction_per_class(model, class_names=None, dataset_dir="dataset"):
    selected_samples = select_one_sample_per_class(
        class_names=class_names,
        dataset_dir=dataset_dir,
    )
    print(f"Selected {len(selected_samples)} samples for prediction and visualization.")
    plot_prediction_samples(selected_samples, model)


### end old section; Funktionen um die Bilder zuzuscheiden und die Labels zu zählen

### begin old section; Funktionen um die Label aus dem Dateinamen zu lesen und den Datensatz zu laden.


def extract_label_from_filename(file_name):
    stem = Path(file_name).stem
    if "__" not in stem:
        raise ValueError(f"Filename does not match '<class>__...': {file_name}")
    label = stem.split("__", 1)[0].strip().lower()
    if not label:
        raise ValueError(f"Empty class label parsed from filename: {file_name}")
    return label


def load_initial_dataset_local(initial_dir):
    """Load images from a single directory and derive labels from filenames.

    Expected filename examples:
    - defectless__.jpg -> class 'defectless'
    - green-quad__(2).jpg -> class 'green-quad'
    """
    initial_dir = Path(initial_dir)
    print(f"Image directory: {initial_dir}")
    samples = []

    if not initial_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {initial_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_path in sorted(initial_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in exts:
            continue
        try:
            class_name = extract_label_from_filename(img_path.name)
            im = Image.open(img_path)
            im = center_crop_resize(im)
            samples.append({"y": class_name, "x": im, "path": str(img_path)})
        except Exception as e:
            print("Skipping image:", img_path.name, "error:", e)

    return samples


def resolve_annotation_image_path(raw_image_value, ann_image_dir):
    p = Path(raw_image_value)
    if p.exists():
        return p

    candidate = Path(ann_image_dir) / Path(raw_image_value).name
    if candidate.exists():
        return candidate

    tail = str(raw_image_value).split("/")[-1]
    candidate2 = Path(ann_image_dir) / tail
    if candidate2.exists():
        return candidate2

    return None


def load_human_annotated_dataset_local(annotation_json_dir, ann_image_dir):
    annotation_json_dir = Path(annotation_json_dir)
    ann_image_dir = Path(ann_image_dir)
    samples = []
    hashlist = []

    if not annotation_json_dir.exists():
        print(
            "Annotation JSON directory not found, returning empty:", annotation_json_dir
        )
        return samples, hashlist

    for json_path in sorted(annotation_json_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data.get("result"):
                continue

            label = data["result"][0]["value"]["choices"][0]
            raw_image = data["task"]["data"]["image"]

            local_image_path = resolve_annotation_image_path(raw_image, ann_image_dir)
            if local_image_path is None:
                print(
                    "Image not found for annotation:",
                    json_path.name,
                    "value:",
                    raw_image,
                )
                continue

            im = Image.open(local_image_path)
            im = center_crop_resize(im)

            samples.append({"y": label, "x": im, "path": str(local_image_path)})
            hashlist.append({"y": label, "obj": str(local_image_path.name)})

        except Exception as e:
            print("Skipping annotation:", json_path, "error:", e)

    return samples, hashlist


### end old section; Funktionen um die Label aus dem Dateinamen zu lesen und den Datensatz zu laden.

### begin old section; Datensatz wird geladen.

initial_dataset = load_initial_dataset_local(initial_dir_w.value)
# print("Initial dataset size:", len(initial_dataset))

human_dataset = []
human_hashlist = []
if use_ann_w.value:
    human_dataset, human_hashlist = load_human_annotated_dataset_local(
        ann_json_dir_w.value, ann_img_dir_w.value
    )

# print("Human annotated dataset size:", len(human_dataset))

if len(initial_dataset) == 0 and len(human_dataset) == 0:
    raise ValueError(
        "No data loaded. Check the Image dir widget and filename pattern '<class>__...'."
    )

all_labels_preview = [s["y"] for s in initial_dataset] + [s["y"] for s in human_dataset]
# print("Class distribution (all loaded):", sample_count_dict(all_labels_preview))

### end old section; Datensatz wird geladen.

### begin old section; Bilder aus dem Datensatz.


def show_samples(samples, n=9, title="Samples"):
    if len(samples) == 0:
        print("No samples to show.")
        return
    n = min(n, len(samples))
    idxs = random.sample(range(len(samples)), n)

    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(12, 4 * rows))
    for i, idx in enumerate(idxs, 1):
        s = samples[idx]
        plt.subplot(rows, cols, i)
        plt.imshow(s["x"])
        plt.title(s["y"])
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def display_samples():
    show_samples(initial_dataset, n=6, title="Initial dataset examples")
    if len(human_dataset) > 0:
        show_samples(human_dataset, n=6, title="Human annotated examples")


### end old section; Bilder aus dem Datensatz.

# Widgets for split and dataloader settings (used in the next split/dataloader cells)
val_split_w = widgets.FloatSlider(
    value=0.2, min=0.05, max=0.4, step=0.05, description="Val split"
)
test_split_w = widgets.FloatSlider(
    value=0.1, min=0.05, max=0.3, step=0.05, description="Test split"
)
batch_w = widgets.IntSlider(
    value=1, min=1, max=64, step=1, description="Batchsize"
)
hflip_w = widgets.FloatSlider(
    value=0.5, min=0.0, max=1.0, step=0.1, description="HFlip"
)
vflip_w = widgets.FloatSlider(
    value=0.5, min=0.0, max=1.0, step=0.1, description="VFlip"
)

dataset_factor_w = widgets.IntSlider(
    value=dataset_factor_default, min=1, max=5, step=1, description="Dataset factor"
)


### set global variables

# x_trainval = None
# x_test = None
# y_trainval = None
# y_test = None
# x_train = None
# x_val = None
# y_train = None
# y_val = None
# class_dict = None

### begin old section; Datensatz wird in den Testdatensatz und des Auswertungsdatensatz aufgeteilt.


def expand_split_samples(samples, factor):
    factor = max(int(factor), 1)
    return list(samples) * factor


def split_data():
    global \
        x_trainval, \
        x_test, \
        y_trainval, \
        y_test, \
        x_train, \
        x_val, \
        y_train, \
        y_val, \
        class_dict, \
        class_names, \
        merged, \
        num_classes
    test_size = float(test_split_w.value)
    val_size = float(val_split_w.value)

    # Merge both sources
    merged = initial_dataset + human_dataset
    random.shuffle(merged)

    labels = [s["y"] for s in merged]

    # Deterministic class mapping for teaching
    class_names = sorted(set(labels))
    class_dict = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    # train+val vs test
    trainval_samples, test_samples, y_trainval_base, y_test_base = train_test_split(
        merged,
        labels,
        test_size=test_size,
        random_state=SEED,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    # train vs val
    train_samples, val_samples, y_train_base, y_val_base = train_test_split(
        trainval_samples,
        y_trainval_base,
        test_size=val_size,
        random_state=SEED,
        stratify=y_trainval_base if len(set(y_trainval_base)) > 1 else None,
    )

    trainval_samples = expand_split_samples(trainval_samples, dataset_factor_w.value)
    train_samples = expand_split_samples(train_samples, dataset_factor_w.value)
    val_samples = expand_split_samples(val_samples, dataset_factor_w.value)
    test_samples = expand_split_samples(test_samples, dataset_factor_w.value)

    x_trainval = [s["x"] for s in trainval_samples]
    y_trainval = [s["y"] for s in trainval_samples]
    x_train = [s["x"] for s in train_samples]
    y_train = [s["y"] for s in train_samples]
    x_val = [s["x"] for s in val_samples]
    y_val = [s["y"] for s in val_samples]
    x_test = [s["x"] for s in test_samples]
    y_test = [s["y"] for s in test_samples]

    print("Train size:", len(x_train), f"(unique before factor: {len(y_train_base)})")
    print("Val size:", len(x_val), f"(unique before factor: {len(y_val_base)})")
    print("Test size:", len(x_test), f"(unique before factor: {len(y_test_base)})")
    print("Classes:", class_dict)
    print("Train class counts:", sample_count_dict(y_train))
    print("Val class counts:", sample_count_dict(y_val))
    print("Test class counts:", sample_count_dict(y_test))
    print("Dataset factor:", dataset_factor_w.value)


### end old section; Datensatz wird in den Testdatensatz und des Auswertungsdatensatz aufgeteilt.

### begin old section; Bilder werden gedreht.


class CustomImageDataset(Dataset):
    def __init__(self, pil_images, labels, class_dict, transform=None):
        self.pil_images = pil_images
        self.labels = labels
        self.class_dict = class_dict
        self.transform = transform

    def __len__(self):
        return len(self.pil_images)

    def __getitem__(self, idx):
        image = self.pil_images[idx]
        label_name = self.labels[idx]
        y = self.class_dict[label_name]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(y, dtype=torch.long)


def get_transforms(hflip=0.5, vflip=0.5):
    train_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.RandomHorizontalFlip(p=hflip),
            transforms.RandomVerticalFlip(p=vflip),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


# train_ds = None
# val_ds = None
# test_ds = None
# train_loader = None
# val_loader = None
# test_loader = None


def create_dataloaders():
    global train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

    train_tfm, eval_tfm = get_transforms(hflip_w.value, vflip_w.value)

    train_ds = CustomImageDataset(x_train, y_train, class_dict, transform=train_tfm)
    val_ds = CustomImageDataset(x_val, y_val, class_dict, transform=eval_tfm)
    test_ds = CustomImageDataset(x_test, y_test, class_dict, transform=eval_tfm)

    print("train_ds size:", len(train_ds))
    print("val_ds size:", len(val_ds))
    print("test_ds size:", len(test_ds))
    # print(" values:", {hflip_w.value}, {vflip_w.value})

    train_loader = DataLoader(
        train_ds, batch_size=batch_w.value, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_w.value, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_w.value, shuffle=False, num_workers=0
    )

    print("Dataloaders ready")


### end old section; Bilder werden gedreht.

### begin old section; Anzeigen von Beispielbildern.

# Visualize transformed samples and compare dataset sizes


def _tensor_to_display_img(tensor_img):
    """Convert normalized CHW tensor back to HWC numpy image for plotting."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor_img.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor_img.dtype).view(3, 1, 1)
    img = tensor_img.detach().cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def show_transform_examples(dataset, n=4):
    if len(dataset) == 0:
        print("Dataset is empty; nothing to visualize.")
        return

    n = min(n, len(dataset))
    idxs = random.sample(range(len(dataset)), n)

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for row, idx in enumerate(idxs):
        original_pil = dataset.pil_images[idx]
        transformed_tensor, _ = dataset[idx]

        # explicit flip examples for teaching
        flipped_h = transforms.functional.horizontal_flip(original_pil)
        flipped_v = transforms.functional.vertical_flip(original_pil)

        axes[row, 0].imshow(original_pil)
        axes[row, 0].set_title("Original")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(flipped_h)
        axes[row, 1].set_title("Horizontal flip")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(flipped_v)
        axes[row, 2].set_title("Vertical flip")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(_tensor_to_display_img(transformed_tensor))
        axes[row, 3].set_title("Random transformed")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.show()


def display_transform_example():
    print("Dataset size comparison")
    print("- Initial dataset:", len(initial_dataset))
    # print("- Human annotated dataset:", len(human_dataset))
    # print("- Merged dataset:", len(merged))
    print("- Train split:", len(train_ds))
    print("- Validation split:", len(val_ds))
    print("- Test split:", len(test_ds))

    if len(initial_dataset) > 0:
        print(
            "- Merged vs Initial ratio:", f"{len(merged) / len(initial_dataset):.2f}x"
        )
        print(
            "- Train vs Initial ratio:", f"{len(train_ds) / len(initial_dataset):.2f}x"
        )

    print("\nShowing train transform examples...")
    show_transform_examples(train_ds, n=4)


### end old section; Anzeigen von Beispielbildern.

### begin old section; Auswahl des Trainingsmodells und Modifizierung der Parameter.

# Widgets for model/training strategy (used in the next model/training cells)
model_name_w = widgets.Dropdown(
    options=["mobilenet_v2", "resnet18", "efficientnet_b0"],
    value="mobilenet_v2",
    description="Model:",
)

training_mode_w = widgets.Dropdown(
    options=[
        ("Feature extraction (freeze backbone)", "feature_extraction"),
        ("Full fine-tuning (train all layers)", "full_finetune"),
    ],
    value="full_finetune",
    description="Mode:",
)

optimizer_name_w = widgets.Dropdown(
    options=["adam", "adamw", "sgd"], value="adam", description="Optimizer:"
)

sgd_momentum_w = widgets.FloatSlider(
    value=0.9,
    min=0.0,
    max=0.99,
    step=0.01,
    description="Momentum (SGD only)",
    style={"description_width": "initial"},
)
lr_w = widgets.FloatLogSlider(
    value=1e-4, base=10, min=-6, max=-2, step=0.1, description="LR"
)
epochs_w = widgets.IntSlider(value=8, min=1, max=50, step=1, description="Epochs")


### end old section; Auswahl des Trainingsmodells und Modifizierung der Parameter.

### begin old section; Erstellen des Modells.


def build_model(num_classes, model_name="mobilenet_v2", freeze_backbone=True):
    if model_name == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1
        )
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=in_features, out_features=num_classes
        )

    elif model_name == "resnet18":
        model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for name, p in model.named_parameters():
                if not name.startswith("fc"):
                    p.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    elif model_name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features=in_features, out_features=num_classes
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model.to(device)


# freeze_backbone = training_mode_w.value == "feature_extraction"
# model = build_model(
#     num_classes=num_classes,
#     model_name=model_name_w.value,
#     freeze_backbone=freeze_backbone,
# )
# print(f"Selected model: {model_name_w.value}")
# print(f"Training mode: {training_mode_w.value}")
# print(model)

### end old section; Erstellen des Modells.


def create_train_button(run_training_func):
    """Create a Train button and Output widget wired to `run_training_func`.

    Returns (button, output_widget).
    """
    output = widgets.Output()

    def on_train_clicked(_):
        with output:
            model, hist, y_true, y_pred = run_training_func(
                epochs=epochs_w.value, lr=lr_w.value
            )

            idx_to_class = {v: k for k, v in class_dict.items()}
            names = [idx_to_class[i] for i in range(len(idx_to_class))]

            print("Classification report:")
            print(
                classification_report(
                    y_true, y_pred, target_names=names, zero_division=0
                )
            )

            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(names))
            plt.xticks(tick_marks, names, rotation=45, ha="right")
            plt.yticks(tick_marks, names)
            plt.tight_layout()
            plt.ylabel("True")
            plt.xlabel("Pred")
            plt.show()

            # Save for later access as utils.trained_model
            global trained_model
            trained_model = model

    train_btn = widgets.Button(description="Train Model", button_style="success")
    train_btn.on_click(on_train_clicked)
    return train_btn, output


######## for evaluation and prediction cells to access the trained model and class names

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def download_and_extract_external_testset(url, target_dir, zip_path, force_redownload=False):
    if not url or "PASTE_RAW_GIST_ZIP_URL_HERE" in url:
        raise ValueError("Bitte EXTERNAL_TESTSET_ZIP_URL mit einer gueltigen RAW-Gist-URL setzen.")

    if force_redownload and target_dir.exists():
        shutil.rmtree(target_dir)
    if force_redownload and zip_path.exists():
        zip_path.unlink()

    if not target_dir.exists():
        print("Lade externes Testset herunter...")
        urllib.request.urlretrieve(url, zip_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        print("Entpackt nach:", target_dir.resolve())
    else:
        print("Nutze vorhandenes entpacktes Testset:", target_dir.resolve())

    return target_dir

def parse_label_from_path(path):
    # 1) Bevorzugt Klassenordner
    parent_label = path.parent.name.strip().lower()
    if parent_label in class_dict:
        return parent_label

    # 2) Fallback auf Dateinamen-Schema class__...
    try:
        file_label = extract_label_from_filename(path.name)
        if file_label in class_dict:
            return file_label
    except Exception:
        pass

    return None

def load_external_samples(root_dir):
    image_paths = [
        p for p in Path(root_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    samples, unknown_labels = [], set()
    for p in sorted(image_paths):
        label = parse_label_from_path(p)
        if label is None:
            unknown_labels.add(p.parent.name)
            continue
        pil_img = Image.open(p).convert("RGB")
        pil_img = center_crop_resize(pil_img)
        samples.append({"x": pil_img, "y": label, "path": str(p)})

    print(f"Gefundene Bilddateien: {len(image_paths)}")
    print(f"Nutzbare Samples (Label passt zu Training): {len(samples)}")
    if unknown_labels:
        print("Uebersprungene Labels/Ordner (nicht im Training):", sorted(unknown_labels))

    if len(samples) == 0:
        raise ValueError("Keine passenden Samples gefunden. Pruefe Klassenlabels im Gist-Testset.")

    return samples

def evaluate_on_external_testset(model, samples):
    eval_tfm = get_transforms(hflip=0.0, vflip=0.0)[1]
    x_ext = [s["x"] for s in samples]
    y_ext = [s["y"] for s in samples]

    ext_ds = CustomImageDataset(x_ext, y_ext, class_dict, transform=eval_tfm)
    ext_loader = DataLoader(
        ext_ds, batch_size=max(1, int(batch_w.value)), shuffle=False, num_workers=0
    )

    all_true, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in ext_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_pred.extend(preds)
            all_true.extend(yb.cpu().numpy().tolist())

    acc = (np.array(all_true) == np.array(all_pred)).mean() * 100.0
    idx_to_class = {v: k for k, v in class_dict.items()}
    names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"\nExternal Test Accuracy: {acc:.2f}%")
    print("Classification report (external):")
    print(classification_report(all_true, all_pred, target_names=names, zero_division=0))

    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("External Test Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45, ha="right")
    plt.yticks(tick_marks, names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.show()

    return acc, all_true, all_pred