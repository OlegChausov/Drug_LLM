# LLM_app/routers/ws_lc.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from LLM_app.chain_lc import stream_answer

router = APIRouter()


@router.websocket("/ws_lc")
async def websocket_endpoint_lc(ws: WebSocket):
    await ws.accept()

    try:
        while True:
            question = await ws.receive_text()
            question = question.strip()

            if question == "/reset":
                await ws.send_text("[контекст сброшен]\n\n")
                continue

            try:
                async for token in stream_answer(question):
                    await ws.send_text(token)
            except WebSocketDisconnect:
                # клиент отвалился — просто выходим
                print(">>> ws_lc: клиент отключился во время стрима")
                return
            except Exception as e:
                # ошибка обработки — пытаемся сообщить, но осторожно
                print(f">>> ws_lc: ошибка стрима: {e}")
                try:
                    await ws.send_text(f"\n[ERROR] {e}")
                except WebSocketDisconnect:
                    return

    except WebSocketDisconnect:
        print(">>> клиент отключился")