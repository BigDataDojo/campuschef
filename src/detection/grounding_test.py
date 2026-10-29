import sys

# IMPORTANT:
# Load torch before GroundingDINO because of the CUDA extension/library issue
# in the current environment.
import torch

import cv2
import numpy as np
from pathlib import Path
from torchvision.ops import nms

# GroundingDINO is installed in editable mode
from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
)


# ============================================================
# 1. SETTINGS
# ============================================================

IMAGE_PATH = "/workspace/main/assets/samples/food2.jpg"

OUTPUT_DIR = Path("/workspace/main/outputs/grounding_dino/grounding_clean_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = (
    "/workspace/GroundingDINO/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)

WEIGHTS_PATH = (
    "/workspace/GroundingDINO/"
    "weights/groundingdino_swint_ogc.pth"
)

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

# NMS removes duplicate boxes for the same class.
NMS_THRESHOLD = 0.50


# ============================================================
# 2. YOUR COOK-AI / YOLO CLASSES
# ============================================================

YOLO_CLASSES = [
    "apple",
    "asparagus",
    "avocado",
    "banana",
    "beef",
    "bell_pepper",
    "bento",
    "blueberries",
    "bottle",
    "bread",
    "broccoli",
    "butter",
    "can",
    "carrot",
    "cauliflower",
    "cheese",
    "cherry",
    "chicken",
    "chicken_breast",
    "chocolate",
    "coffee",
    "corn",
    "cream-cheese",
    "cucumber",
    "dates",
    "egg",
    "eggplant",
    "eggs",
    "energy_drink",
    "fish",
    "flour",
    "garlic",
    "ginger",
    "goat_cheese",
    "grapefruit",
    "grapes",
    "grated_cheese",
    "green-chillies",
    "green_beans",
    "ground_beef",
    "guacamole",
    "ham",
    "heavy_cream",
    "humus",
    "juice",
    "ketchup",
    "kiwi",
    "leek",
    "lemon",
    "lettuce",
    "lime",
    "mango",
    "marmelade",
    "mayonaise",
    "milk",
    "mint",
    "mushrooms",
    "mustard",
    "nuts",
    "olives",
    "onion",
    "orange",
    "pak_choi",
    "parsley",
    "peach",
    "pear",
    "peas",
    "pickles",
    "pineapple",
    "plasticsaveholder",
    "pomegrante",
    "pot",
    "potato",
    "potatoes",
    "pudding",
    "radish",
    "red_cabbage",
    "red_grapes",
    "red_onion",
    "rice_ball",
    "salad",
    "salami",
    "sandwich",
    "sausage",
    "shrimp",
    "smoothie",
    "spinach",
    "spring_onion",
    "strawberries",
    "sugar",
    "sweet_potato",
    "tangerine",
    "tea_a",
    "tea_i",
    "tomato",
    "tomato_sauce",
    "tortillas",
    "turkey",
    "watermelon",
    "yogurt",
    "zucchini",
]


# ============================================================
# 3. CONVERT YOLO CLASS NAME -> NATURAL LANGUAGE
# ============================================================

# Grounding DINO works with natural-language text.
# Keep the original YOLO name for the final output, but use a cleaner
# natural-language prompt for DINO.

PROMPT_MAP = {
    "bell_pepper": "bell pepper",
    "chicken_breast": "chicken breast",
    "cream-cheese": "cream cheese",
    "energy_drink": "energy drink",
    "goat_cheese": "goat cheese",
    "grated_cheese": "grated cheese",
    "green-chillies": "green chillies",
    "green_beans": "green beans",
    "ground_beef": "ground beef",
    "heavy_cream": "heavy cream",
    "pak_choi": "pak choi",
    "red_cabbage": "red cabbage",
    "red_grapes": "red grapes",
    "red_onion": "red onion",
    "rice_ball": "rice ball",
    "spring_onion": "spring onion",
    "sweet_potato": "sweet potato",
    "tomato_sauce": "tomato sauce",

    # Dataset spelling -> natural spelling
    "pomegrante": "pomegranate",
    "marmelade": "marmalade",
    "mayonaise": "mayonnaise",

    # Dataset-specific names.
    # These may need adjustment depending on what the original
    # Cook-AI dataset intended them to mean.
    "plasticsaveholder": "plastic food container",
    "tea_a": "tea",
    "tea_i": "tea",
}


def get_prompt(class_name):
    return PROMPT_MAP.get(
        class_name,
        class_name.replace("_", " ").replace("-", " ")
    )


# ============================================================
# 4. LOAD GROUNDING DINO
# ============================================================

print("=" * 60)
print("Loading Grounding DINO...")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

model = load_model(
    CONFIG_PATH,
    WEIGHTS_PATH,
    device=device,
)

print("Grounding DINO loaded.")
print()


# ============================================================
# 5. LOAD IMAGE
# ============================================================

image_source, image = load_image(IMAGE_PATH)

height, width = image_source.shape[:2]

print("Image:", IMAGE_PATH)
print("Size:", width, "x", height)
print()


# ============================================================
# 6. RUN EACH CLASS SEPARATELY
# ============================================================

detections = []

print("=" * 60)
print("Running detection")
print("=" * 60)

for class_id, class_name in enumerate(YOLO_CLASSES):

    prompt = get_prompt(class_name)

    print(
        f"[{class_id + 1:03d}/{len(YOLO_CLASSES)}] "
        f"{class_name:<20} -> '{prompt}'"
    )

    try:
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device,
        )

    except Exception as e:
        print(f"    ERROR: {e}")
        continue

    if len(boxes) == 0:
        continue

    # GroundingDINO boxes are normalized:
    # center_x, center_y, width, height
    for box, score in zip(boxes, logits):

        cx, cy, bw, bh = box.tolist()

        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height

        # Keep coordinates inside image
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "prompt": prompt,
                "score": float(score),
                "box": [x1, y1, x2, y2],
            }
        )


print()
print("Raw detections:", len(detections))


# ============================================================
# 7. NMS PER CLASS
# ============================================================

final_detections = []

for class_id, class_name in enumerate(YOLO_CLASSES):

    class_detections = [
        d for d in detections
        if d["class_id"] == class_id
    ]

    if not class_detections:
        continue

    boxes_tensor = torch.tensor(
        [d["box"] for d in class_detections],
        dtype=torch.float32,
    )

    scores_tensor = torch.tensor(
        [d["score"] for d in class_detections],
        dtype=torch.float32,
    )

    keep = nms(
        boxes_tensor,
        scores_tensor,
        NMS_THRESHOLD,
    )

    for index in keep.tolist():
        final_detections.append(
            class_detections[index]
        )


# Sort highest confidence first
final_detections.sort(
    key=lambda d: d["score"],
    reverse=True,
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL DETECTIONS")
print("=" * 60)

if not final_detections:
    print("No objects detected.")

for detection in final_detections:

    print(
        f"{detection['class_name']:<20} "
        f"{detection['score']:.3f}"
    )


# ============================================================
# 9. DRAW BOUNDING BOXES
# ============================================================

result_image = image_source.copy()

for detection in final_detections:

    x1, y1, x2, y2 = map(
        int,
        detection["box"]
    )

    score = detection["score"]
    class_name = detection["class_name"]

    # Generate deterministic color based on class ID
    class_id = detection["class_id"]

    color = (
        int((37 * class_id) % 255),
        int((17 * class_id + 80) % 255),
        int((29 * class_id + 160) % 255),
    )

    cv2.rectangle(
        result_image,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    label = f"{class_name} {score:.2f}"

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )

    label_y = max(y1, text_height + 5)

    cv2.rectangle(
        result_image,
        (x1, label_y - text_height - 5),
        (x1 + text_width + 5, label_y + baseline),
        color,
        -1,
    )

    cv2.putText(
        result_image,
        label,
        (x1 + 2, label_y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# 10. SAVE IMAGE
# ============================================================

output_path = OUTPUT_DIR / "food2_grounding_clean.jpg"

# load_image() gives RGB; OpenCV expects BGR when writing.
result_bgr = cv2.cvtColor(
    result_image,
    cv2.COLOR_RGB2BGR,
)

cv2.imwrite(
    str(output_path),
    result_bgr,
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print("Saved to:")
print(output_path)