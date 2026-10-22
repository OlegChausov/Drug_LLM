
        const ws = new WebSocket(`ws://${location.host}/ws_lc`);

        let lastLLMBlock = null;
        let llmBuffer = "";
        let renderScheduled = false;
        let isGenerating = false;
        let generationTimer = null;

        const chat = document.getElementById("chat");
        const promptInput = document.getElementById("prompt");
        const sendBtn = document.getElementById("send-btn");

        // ========== РЕНДЕР MARKDOWN ==========
        function scheduleRender() {
            if (renderScheduled) return;
            renderScheduled = true;
            setTimeout(() => {
                if (lastLLMBlock) {
                    lastLLMBlock.innerHTML = marked.parse(llmBuffer);
                    chat.scrollTop = chat.scrollHeight;
                }
                renderScheduled = false;
            }, 100);
        }

        // ========== WEBSOCKET ==========
        ws.onmessage = (event) => {
            if (!lastLLMBlock) {
                lastLLMBlock = document.createElement("div");
                lastLLMBlock.className = "msg llm";
                chat.appendChild(lastLLMBlock);
                llmBuffer = "";

                // индикатор «печатает»
                lastLLMBlock.innerHTML = `
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>`;
            }

            llmBuffer += event.data;
            scheduleRender();

            // обновляем таймер завершения
            if (generationTimer) clearTimeout(generationTimer);
            generationTimer = setTimeout(finishGeneration, 1500);
        };

        ws.onopen = () => console.log("ws_lc connected");

        ws.onclose = () => {
            console.log("ws_lc closed");
            sendBtn.disabled = true;
            sendBtn.textContent = "Offline";
            promptInput.disabled = true;
        };

        ws.onerror = (e) => console.error("ws_lc error", e);

        // ========== КНОПКА «КОПИРОВАТЬ» ==========
        function addCopyButton(block) {
            const btn = document.createElement("button");
            btn.className = "copy-btn";
            btn.textContent = "📋 Копировать";
            btn.onclick = (e) => {
                e.stopPropagation();
                navigator.clipboard.writeText(llmBuffer).then(() => {
                    btn.textContent = "✓ Скопировано";
                    btn.classList.add("copied");
                    setTimeout(() => {
                        btn.textContent = "📋 Копировать";
                        btn.classList.remove("copied");
                    }, 1500);
                });
            };
            block.appendChild(btn);
        }

        // ========== ЗАВЕРШЕНИЕ ГЕНЕРАЦИИ ==========
        function finishGeneration() {
            isGenerating = false;
            sendBtn.disabled = false;
            sendBtn.textContent = "Отправить";
            if (lastLLMBlock && llmBuffer.trim()) {
                addCopyButton(lastLLMBlock);
            }
        }

        // ========== ОТПРАВКА ==========
        function sendMessage() {
            const prompt = promptInput.value.trim();
            if (!prompt || isGenerating) return;

            ws.send(prompt);

            // блок пользователя
            const you = document.createElement("div");
            you.className = "msg you";
            you.textContent = prompt;
            chat.appendChild(you);

            // сброс состояния
            lastLLMBlock = null;
            llmBuffer = "";
            isGenerating = true;

            // очистка поля
            promptInput.value = "";
            autoResize();
            chat.scrollTop = chat.scrollHeight;

            // блокировка кнопки
            sendBtn.disabled = true;
            sendBtn.textContent = "Генерация…";
        }

        // ========== ENTER / SHIFT+ENTER ==========
        promptInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // ========== АВТО-РАЗМЕР TEXTAREA ==========
        function autoResize() {
            promptInput.style.height = "auto";
            promptInput.style.height = Math.min(promptInput.scrollHeight, 160) + "px";
        }
        promptInput.addEventListener("input", autoResize);

        // ========== FOCUS ==========
        promptInput.focus();
  