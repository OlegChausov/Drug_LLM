# LLM_app/chain_lc.py

"""
LangChain-версия RAG-конвейера с памятью диалога.

Наружу отдаёт:
  - stream_answer(question, session_id) — async-генератор токенов
  - clear_history(session_id) — сброс истории сессии
"""

import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from LLM_app.rag import build_context_async, format_context_for_llm

load_dotenv()

MODEL = os.getenv("MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# --- Модель ---
llm = ChatOllama(
    model=MODEL,
    base_url=OLLAMA_BASE_URL,
    # reasoning=False,
    num_predict=800,
)

# --- Промпт с MessagesPlaceholder ---

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """Ты — медицинский ассистент.
        Сначала используй приведённый ниже контекст из инструкций.
        Если в контексте нет ответа — можешь ответить из собственных знаний,
        но обязательно уточни, что это не заменяет консультацию врача или работника аптеки.
        При тяжёлой депрессии или суицидальных мыслях — обязательно направь к врачу."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{context}\n\n=== ВОПРОС ===\n{question}\n\n=== ОТВЕТ ==="),
])
# --- Базовая цепочка ---
base_chain = prompt | llm | StrOutputParser()

# --- Хранилище историй: session_id → InMemoryChatMessageHistory ---
_history_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Возвращает (или создаёт) историю для сессии."""
    if session_id not in _history_store:
        _history_store[session_id] = InMemoryChatMessageHistory()
    return _history_store[session_id]


def clear_history(session_id: str) -> None:
    """Очищает историю сессии."""
    _history_store.pop(session_id, None)


# --- Цепочка с автоматической памятью ---
chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


async def stream_answer(question: str, session_id: str = "default"):
    # 1. RAG-контекст (с fallback на случай ошибки)
    try:
        ctx = await build_context_async(question)
        context_text = format_context_for_llm(ctx, question)
        print(
            f">>> RAG: chunks={len(ctx['chunks'])}, "
            f"entities={ctx.get('entities', [])}, "
            f"edges={len(ctx['edges'])}"
        )
    except Exception as e:
        print(f">>> RAG ERROR: {e}")
        context_text = ""   # fallback — без контекста

    # 2. Стриминг с автоматической памятью
    async for token in chain.astream(
        {"context": context_text, "question": question},
        config={"configurable": {"session_id": session_id}},
    ):
        yield token