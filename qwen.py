import os

# Ограничиваем видимые GPU только 4 и 5
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, re
from tqdm import tqdm

# === Параметры ===
INPUT_CSV  = "test.csv"
OUTPUT_CSV = "out.csv"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
BATCH_SIZE = 4  # подберите под свою память

# === 1) Датасет ===
df    = pd.read_csv(INPUT_CSV)
texts = df["Description"].tolist()
print(f"✅ Загружено строк: {len(texts)}")

# === 2) Модель ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",       # теперь auto распознает только 2 GPU (они стали 0 и 1)
    trust_remote_code=True
).eval()
device = next(model.parameters()).device

# === 3) Системный промпт ===
SYSTEM_PROMPT = """Ты — классификатор медицинских данных. Анализируй текст и определяй тип визита.

### Правила:
1. Primary (первичный):
   - Первый визит
   - Первичная консультация
   - Впервые установлен диагноз
   - Первичное обследование

2. Follow-up (повторный):
   - Контрольный осмотр
   - Повторный визит
   - Динамическое наблюдение
   - Коррекция лечения
   - Сравнение с предыдущим состоянием

### Формат ответа:
{"text": "исходный текст", "category": "Primary|Follow-up"}

Никаких пояснений, только JSON!"""

# === 4) Ключевые слова для эвристик ===
primary_keywords = [
    "первичн", "впервые", "первый раз", "установлен диагноз", "первичный", "впервый"
]

followup_keywords = [
    'повторно', 'фоллоуап', 'контроль', 'динамика', 'без изменений', 'стабильно', 'через',
    'сравнительно', 'сравнение', 'по сравнению', 'повторное обследование', 'повторная рентгенография',
    'на фоне предыдущего', 'улучшение', 'ухудшение', 'прогрессия', 'регрессия', 'стабилизация',
    'изменение по сравнению', 'результаты предыдущего', 'повторный снимок', 'повторное исследование',
    'в динамике', 'после лечения', 'после терапии', 'повторная оценка', 'повторный осмотр',
    'контрольное исследование', 'контрольный снимок', 'контрольная рентгенография', 'стабильное состояние',
    'отсутствие прогрессирования', 'отсутствие изменений', 'без существенных изменений', 'без новых очагов',
    'без признаков', 'состояние после', 'после операции', 'после вмешательства', 'после лечения'
]

# === 5) Вспомогательные функции ===
def gpu_mem():
    return "; ".join(
        f"GPU{i}:{torch.cuda.memory_allocated(i)/1e9:.1f}/{torch.cuda.memory_reserved(i)/1e9:.1f} GB"
        for i in range(torch.cuda.device_count())
    )

def build_prompt(text: str) -> str:
    "Возвращает готовую строку-промпт по чат‐шаблону"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Текст: {text}"}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

def classify_batch(batch_texts):
    # 1) Собираем строковые промпты
    prompts = [build_prompt(t) for t in batch_texts]

    # 2) Токенизируем обычным batch-encode
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 3) Генерируем
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4) Обрезаем префикс, декодируем
    gen     = out[:, input_ids.shape[1]:]
    replies = tokenizer.batch_decode(gen, skip_special_tokens=True)

    # 5) Парсим JSON + эвристика по ключевым словам
    cats = []
    for orig, rep in zip(batch_texts, replies):
        cat = None
        m = re.search(r"\{.*?\}", rep, flags=re.S)
        if m:
            try:
                cat = json.loads(m.group(0).replace("“", '"').replace("”", '"')).get("category")
            except json.JSONDecodeError:
                cat = None

        if not cat:
            lw = orig.lower()
            if any(k in lw for k in primary_keywords):
                cat = "Primary"
            elif any(k in lw for k in followup_keywords):
                cat = "Follow-up"
            else:
                cat = "Follow-up"
        cats.append(cat)
    return cats

# === 6) Основной проход ===
preds = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batches"):
    batch = texts[i : i + BATCH_SIZE]
    preds.extend(classify_batch(batch))
    tqdm.write(gpu_mem())

# === 7) Сохраняем результат ===
df["predicted_category"] = preds
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"✅ Готово! Результат сохранён в {OUTPUT_CSV}")
