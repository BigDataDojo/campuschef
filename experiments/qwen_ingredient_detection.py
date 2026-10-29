import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

model_name = "Qwen/Qwen3.5-4B"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16
)

image = Image.open("/workspace/main/assets/samples/food2.jpg").convert("RGB")

prompt = """
画像に写っている食材だけを検出してください。

ルール:
- 日本語だけで答える。
- 料理提案はしない。
- 栄養情報は書かない。
- 調理手順は書かない。
- 分析や説明は書かない。
- 食材名だけを書く。
- 1行に1つの食材を書く。
- 不確実なものは「不明」と書く。

出力形式:

【検出した食材】
-
-
-
"""

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """
You are an ingredient detection system.
Answer only in Japanese.
Return only visible food ingredients.
Do not write reasoning or explanations.
"""
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=3000,
    do_sample=False,
    repetition_penalty=1.2
)

answer = processor.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
    
)
if "</think>" in answer:
    answer = answer.split("</think>")[-1].strip()


print(answer)