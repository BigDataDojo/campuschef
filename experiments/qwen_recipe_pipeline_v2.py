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

def ask_qwen(messages, max_new_tokens=3000):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2
        )

    answer = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    if "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()

    return answer.strip()


def detect_ingredients(image_path):
    image = Image.open(image_path).convert("RGB")

    prompt = """
画像に写っている食材だけを検出してください。

ルール:
- 日本語だけで答える
- 食材名だけを書く
- 1行に1つの食材を書く
- 料理名、説明、栄養情報、調理手順は書かない。
- 不確実なものは「不明」と書く。

出力形式:
食材:
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
                    "text": "You are an ingredient detection system. Answer only in Japanese."
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

    return ask_qwen(messages, max_new_tokens=3000)


def make_recipe(ingredients):
    prompt = f"""
次の食材を使って、学生向けの簡単なレシピを日本語で作ってください。

食材:
{ingredients}

重要ルール:
- 日本語だけで答える
- Thinking Processを書かない
- 考え方や分析を書かない
- 最終回答だけを書く
- すべての食材を使う必要はない
- 安く作れる料理にする
- 初心者でも作れるようにする
- 足りない調味料があれば「あるとよいもの」に書く
- 分量はだいたいでよい

出力形式:
【料理名】

【使う食材】

【あるとよいもの】

【作り方】
1.
2.
3.
4.
5.

【ポイント】
"""

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
You are a recipe assistant.
Answer only in Japanese.
Do not write reasoning.
Do not write Thinking Process.
Output only the final recipe.
"""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    return ask_qwen(messages, max_new_tokens=5000)

image_path = "/workspace/main/assets/samples/food2.jpg"

ingredients = detect_ingredients(image_path)
recipe = make_recipe(ingredients)

print("===== 検出した食材 =====")
print(ingredients)

print("\n===== 提案レシピ =====")
print(recipe)