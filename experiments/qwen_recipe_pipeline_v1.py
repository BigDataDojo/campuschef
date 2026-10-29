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

def clean_answer(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text.strip()

def generate_answer(messages, max_tokens=500):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        repetition_penalty=1.2
    )

    answer = processor.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    return clean_answer(answer)

# Step 1: Image -> Ingredients
image = Image.open("/workspace/main/assets/samples/food2.jpg").convert("RGB")

ingredient_prompt = """
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

ingredient_messages = [
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
            {"type": "text", "text": ingredient_prompt}
        ]
    }
]

ingredients_answer = generate_answer(ingredient_messages, max_tokens=500)

print("=== 検出した食材 ===")
print(ingredients_answer)


# Step 2: Ingredients -> Recipe
recipe_prompt = f"""
以下の食材を使って、学生向けの安くて簡単な料理を1つ提案してください。

食材:
{ingredients_answer}

ルール:
- 日本語だけで答える。
- 分析や説明は書かない。
- 最終回答だけを書く。
- 料理は1つだけ提案する。
- 調理時間、予算、栄養情報も書く。

出力形式:

【おすすめレシピ】

【使用する食材】

【不足食材】

【調理時間】

【予算目安】

【栄養情報】
- カロリー:
- タンパク質:
- 脂質:
- 炭水化物:

【調理手順】
1.
2.
3.
"""

recipe_messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """
You are CampusChef.
Answer only in Japanese.
Recommend one cheap and simple recipe for university students.
Do not write reasoning or explanations.
"""
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": recipe_prompt}
        ]
    }
]

recipe_answer = generate_answer(recipe_messages, max_tokens=400)

print("\n=== レシピ提案 ===")
print(recipe_answer)