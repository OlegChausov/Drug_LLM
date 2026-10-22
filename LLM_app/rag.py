# rag.py

"""
RAG-модуль: Qdrant + Neo4j + Ollama.

Функции:
  - retrieve_chunks(question)          → топ-5 чанков из Qdrant
  - extract_drugs_from_chunks(chunks)  → список препаратов из payload
  - get_graph_context(english_names)   → связи из Neo4j по english_name
  - build_context(question)            → синхронная сборка (старая)
  - build_context_async(question)      → гибридная сборка (Qdrant + LLM-NER + Neo4j)
"""

import os
import asyncio
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

load_dotenv()

# --- Настройки (из .env, с дефолтами) ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "drugs")
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "120"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j_for_drugs")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-large")

TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "5"))
MAX_GRAPH_EDGES = int(os.getenv("MAX_GRAPH_EDGES", "40"))


# --- Ленивая инициализация ---
_embed_model = None
_qdrant = None
_neo4j_driver = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("📦 Загрузка модели эмбеддингов...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT)
    return _qdrant


def get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _neo4j_driver


# ---------------------------------------------------------------------------
# 1. Векторный поиск в Qdrant
# ---------------------------------------------------------------------------
def retrieve_chunks(question: str, top_k: int = TOP_K_CHUNKS):
    """
    Возвращает список чанков из Qdrant, отсортированных по релевантности.
    Каждый чанк: {"document", "section", "text", "score"}
    """
    model = get_embed_model()
    client = get_qdrant()

    query_vector = model.encode(question).tolist()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points

    chunks = []
    for hit in results:
        p = hit.payload or {}
        chunks.append({
            "document": p.get("document", ""),
            "section": p.get("section", ""),
            "text": p.get("text", ""),
            "score": round(float(hit.score), 4),
        })

    return chunks


# ---------------------------------------------------------------------------
# 2. Извлекаем препараты из найденных чанков (для старого build_context)
# ---------------------------------------------------------------------------
def extract_drugs_from_chunks(chunks):
    """Возвращает список уникальных document'ов из чанков."""
    return list({c["document"] for c in chunks if c["document"]})


# ---------------------------------------------------------------------------
# 3. Графовый контекст из Neo4j (по english_name)
# ---------------------------------------------------------------------------
def get_graph_context(english_names, limit: int = MAX_GRAPH_EDGES):
    """
    Ищем связи в Neo4j по английским именам (english_name).
    Возвращает список словарей:
      {"source", "source_en", "relation", "target", "target_en", "target_type"}
    """
    if not english_names:
        return []

    driver = get_neo4j()
    names = [n.strip().lower() for n in english_names if n]

    query = """
    MATCH (n)-[r]-(other)
    WHERE toLower(n.english_name) IN $names
    RETURN DISTINCT
           coalesce(n.name, '')             AS source,
           coalesce(n.english_name, '')     AS source_en,
           coalesce(r.type, type(r))        AS relation,
           coalesce(other.name, '')         AS target,
           coalesce(other.english_name, '') AS target_en,
           coalesce(other.type, '')         AS target_type
    LIMIT $limit
    """

    with driver.session() as session:
        result = session.run(query, names=names, limit=limit)
        return [dict(rec) for rec in result]


# ---------------------------------------------------------------------------
# 4a. Синхронная сборка контекста (старая — для ws.py)
# ---------------------------------------------------------------------------
def build_context(question: str):
    """
    Полный RAG-контекст (синхронно):
      - чанки из Qdrant,
      - связи из Neo4j (по русским именам из чанков).
    """
    chunks = retrieve_chunks(question)
    drugs = extract_drugs_from_chunks(chunks)
    edges = get_graph_context(drugs)

    return {
        "chunks": chunks,
        "drugs": drugs,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# 4b. Гибридная async сборка (новая — для chain_lc)
# ---------------------------------------------------------------------------
async def build_context_async(question: str):
    """
    Гибридная сборка контекста:
      - ветка A: Qdrant (retrieve_chunks) — параллельно
      - ветка B: LLM-NER (extract_entities) → Neo4j (get_graph_context)
    Ветки работают параллельно.
    """
    from LLM_app.ner import extract_entities  # импорт внутри — чтобы избежать циклов

    # Ветка A: Qdrant — в отдельном потоке (синхронный клиент)
    chunks_task = asyncio.to_thread(retrieve_chunks, question)

    # Ветка B: LLM-NER — уже async
    entities_task = extract_entities(question)

    # Ждём обе
    chunks, entities = await asyncio.gather(chunks_task, entities_task)

    # Граф — по английским именам из NER
    edges = await asyncio.to_thread(get_graph_context, entities)

    return {
        "chunks": chunks,
        "entities": entities,   # что извлёк NER
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# 5. Формирование промпта для LLM
# ---------------------------------------------------------------------------
def format_context_for_llm(context: dict, question: str) -> str:
    """
    Превращает RAG-контекст в текстовый промпт для Ollama/Qwen.
    Поддерживает оба формата: старый (drugs) и новый (entities).
    """
    lines = []
    lines.append("Ты — медицинский ассистент. Отвечай на вопрос пользователя,")
    lines.append("используя ТОЛЬКО приведённый ниже контекст из инструкций.")
    lines.append("Если ответа в контексте нет — так и скажи.")
    lines.append("")
    lines.append("=== ФРАГМЕНТЫ ИНСТРУКЦИЙ ===")

    for i, c in enumerate(context.get("chunks", []), 1):
        doc = c.get("document", "")
        sec = c.get("section", "")
        text = c.get("text", "")[:800]
        lines.append(f"\n[Фрагмент {i}] Препарат: {doc} | Раздел: {sec}")
        lines.append(text)

    edges = context.get("edges", [])
    if edges:
        lines.append("")
        lines.append("=== СВЯЗИ ИЗ ГРАФА ЗНАНИЙ ===")
        for e in edges[:30]:
            source = e.get("source", "")
            relation = e.get("relation", "")
            target = e.get("target", "")
            target_type = e.get("target_type", "")
            lines.append(
                f"{source} -[{relation}]-> {target} ({target_type})"
            )

    lines.append("")
    lines.append("=== ВОПРОС ===")
    lines.append(question)
    lines.append("")
    lines.append("=== ОТВЕТ ===")

    return "\n".join(lines)