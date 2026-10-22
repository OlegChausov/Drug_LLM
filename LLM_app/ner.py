# LLM_app/ner.py

import os
import json
from langchain_ollama import ChatOllama

MODEL = os.getenv("MODEL_NER", "qwen3:4b")

llm_ner = ChatOllama(
    model=MODEL,
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    format="json",       # Ollama вернёт валидный JSON
)


NER_PROMPT = """\
Извлеки из вопроса медицинские сущности и переведи их в английские канонические названия.

Правила:
- Используй английские названия в единственном числе, lowercase.
- Фармакологические термины на латинице (например: paroxetine, alcohol, serotonin).
- НЕ переводи симптомы буквально — используй медицинские термины (например: "болит голова" → "headache").
- Пустой список — валидный ответ.

Примеры:
- "Можно ли пить алкоголь с пароксетином?" → ["alcohol", "paroxetine"]
- "Какие побочки у сертралина?" → ["sertraline"]
- "Что такое депрессия?" → ["depression"]

Вопрос: {question}

JSON: {{"entities": [...]}}
"""


async def extract_entities(question: str) -> list[str]:
    """
    Извлекает сущности из вопроса и возвращает их английские канонические имена.
    """
    prompt = NER_PROMPT.format(question=question)
    
    try:
        resp = await llm_ner.ainvoke(prompt)
        data = json.loads(resp.content)
        entities = data.get("entities", [])
        # нормализуем: lower, strip
        entities = [e.strip().lower() for e in entities if e]
        return entities
    except Exception as e:
        print(f">>> NER ERROR: {e}")
        return []