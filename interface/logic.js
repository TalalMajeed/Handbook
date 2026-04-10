const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const promptInput = document.getElementById("promptInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const statusDot = document.getElementById("statusDot");

let activeSource = null;

function scrollToBottom() {
    chatLog.scrollTop = chatLog.scrollHeight;
}

function createMessage(role, content = "") {
    const el = document.createElement("div");
    el.className = `message ${role}`;
    el.textContent = content;
    chatLog.appendChild(el);
    scrollToBottom();
    return el;
}

function createTypingIndicator() {
    const el = document.createElement("div");
    el.className = "message assistant";
    el.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    chatLog.appendChild(el);
    scrollToBottom();
    return el;
}

function setStatus(state, title) {
    statusDot.className = `status-dot status-${state}`;
    statusDot.title = title;
}

function setBusy(isBusy) {
    sendBtn.disabled = isBusy;
    clearBtn.disabled = isBusy;
    promptInput.disabled = isBusy;
}

function autoResize() {
    promptInput.style.height = "auto";
    promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`;
}

function stopStream() {
    if (activeSource) {
        activeSource.close();
        activeSource = null;
    }
}

function handleChunk(target, data) {
    target.textContent += data ?? "";
    scrollToBottom();
}

chatForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    stopStream();
    createMessage("user", prompt);
    promptInput.value = "";
    autoResize();

    const assistantMessage = createMessage("assistant", "");
    const typing = createTypingIndicator();
    setBusy(true);
    setStatus("connecting", "Connecting");

    const source = new EventSource(`/generate?prompt=${encodeURIComponent(prompt)}`);
    activeSource = source;

    source.addEventListener("start", () => {
        typing.remove();
        setStatus("streaming", "Streaming");
    });

    source.addEventListener("delta", (event) => {
        typing.remove();
        handleChunk(assistantMessage, event.data);
        setStatus("streaming", "Streaming");
    });

    source.addEventListener("done", () => {
        typing.remove();
        source.close();
        activeSource = null;
        setStatus("ready", "Ready");
        setBusy(false);
    });

    source.addEventListener("error", () => {
        typing.remove();
        source.close();
        activeSource = null;
        if (!assistantMessage.textContent) {
            assistantMessage.textContent = "Connection failed.";
        }
        setStatus("error", "Connection failed");
        setBusy(false);
    });
});

promptInput.addEventListener("input", autoResize);
promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});

clearBtn.addEventListener("click", () => {
    stopStream();
    chatLog.innerHTML = "";
    setBusy(false);
    setStatus("ready", "Ready");
    promptInput.focus();
});

createMessage("system", "Ask anything to start.");
setStatus("ready", "Ready");
autoResize();
