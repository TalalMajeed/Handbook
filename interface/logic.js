// ── DOM refs ──────────────────────────────────────────────────────────────────
const chatLog        = document.getElementById("chatLog");
const chatForm       = document.getElementById("chatForm");
const promptInput    = document.getElementById("promptInput");
const sendBtn        = document.getElementById("sendBtn");
const clearBtn       = document.getElementById("clearBtn");
const statusDot      = document.getElementById("statusDot");
const evidenceList   = document.getElementById("evidenceList");
const evidenceMethod = document.getElementById("evidenceMethod");
const chunkModal     = document.getElementById("chunkModal");
const modalClose     = document.getElementById("modalClose");
const modalMeta      = document.getElementById("modalMeta");
const modalBody      = document.getElementById("modalBody");

let activeSource = null;

// ── Utility ───────────────────────────────────────────────────────────────────

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function scrollBottom() { chatLog.scrollTop = chatLog.scrollHeight; }

function setStatus(state, title) {
    statusDot.className = `status-dot status-${state}`;
    statusDot.title = title;
}

function setBusy(b) {
    sendBtn.disabled = b;
    clearBtn.disabled = b;
    promptInput.disabled = b;
}

function autoResize() {
    promptInput.style.height = "auto";
    promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`;
}

function stopStream() {
    if (activeSource) { activeSource.close(); activeSource = null; }
}

// ── Chat messages ─────────────────────────────────────────────────────────────

function createMessage(role, content = "") {
    const el = document.createElement("div");
    el.className = `message ${role}`;
    el.textContent = content;
    chatLog.appendChild(el);
    scrollBottom();
    return el;
}

function createTyping() {
    const el = document.createElement("div");
    el.className = "message assistant";
    el.innerHTML = '<span class="typing-dot"><span></span><span></span><span></span></span>';
    chatLog.appendChild(el);
    scrollBottom();
    return el;
}

// ── Modal ─────────────────────────────────────────────────────────────────────

function openModal(chunk) {
    // Populate meta badges
    const srcLabel = (chunk.source || "handbook").replace(/\.pdf$/i, "");
    modalMeta.innerHTML = `
        <span class="chunk-rank">#${chunk.rank}</span>
        <span class="chunk-source-badge" title="${escapeHtml(chunk.source || '')}">
            ${escapeHtml(srcLabel)}
        </span>
        <span class="chunk-page-badge">Page ${escapeHtml(String(chunk.page))}</span>
        <span class="method-badge">${escapeHtml(chunk.method || "hybrid")}</span>
        <span class="chunk-score">score: ${chunk.score}</span>
    `;
    // Full text — no truncation
    modalBody.textContent = chunk.fullText || chunk.text || "";
    chunkModal.classList.add("open");
    document.body.style.overflow = "hidden";
}

function closeModal() {
    chunkModal.classList.remove("open");
    document.body.style.overflow = "";
}

modalClose.addEventListener("click", closeModal);
chunkModal.addEventListener("click", (e) => {
    if (e.target === chunkModal) closeModal();
});
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
});

// ── Evidence panel ────────────────────────────────────────────────────────────

function clearEvidence() {
    evidenceList.innerHTML = '<p class="evidence-placeholder">Retrieved chunks will appear here after you ask a question.</p>';
    evidenceMethod.textContent = "";
}

function renderEvidence(chunks) {
    evidenceList.innerHTML = "";

    if (!chunks || chunks.length === 0) {
        evidenceList.innerHTML = '<p class="evidence-placeholder">No chunks retrieved.</p>';
        return;
    }

    if (chunks[0]?.method) {
        evidenceMethod.textContent = chunks[0].method;
    }

    chunks.forEach(chunk => {
        const card = document.createElement("div");
        card.className = "chunk-card";
        card.setAttribute("role", "button");
        card.setAttribute("tabindex", "0");
        card.setAttribute("title", "Click to view full text");

        const srcLabel    = (chunk.source || "handbook").replace(/\.pdf$/i, "");
        const previewText = chunk.preview || (chunk.text || "").slice(0, 260);
        const needsMore   = (chunk.text || "").length > 260;

        card.innerHTML = `
            <div class="chunk-meta-row">
                <span class="chunk-rank">#${escapeHtml(String(chunk.rank))}</span>
                <span class="chunk-source-badge" title="${escapeHtml(chunk.source || '')}">${escapeHtml(srcLabel)}</span>
                <span class="chunk-page-badge">p.${escapeHtml(String(chunk.page))}</span>
                <span class="chunk-score">score: ${chunk.score}</span>
            </div>
            <div class="chunk-text-preview">${escapeHtml(previewText)}${needsMore ? "…" : ""}</div>
            <div class="chunk-open-hint">&#8599; Click to read full text</div>
        `;

        // Store full text on the object for the modal
        chunk.fullText = chunk.text || "";

        card.addEventListener("click",  () => openModal(chunk));
        card.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") openModal(chunk); });

        evidenceList.appendChild(card);
    });
}

// ── Stream handler ────────────────────────────────────────────────────────────

chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    stopStream();
    clearEvidence();

    createMessage("user", prompt);
    promptInput.value = "";
    autoResize();

    const assistantMsg = createMessage("assistant", "");
    const typing       = createTyping();
    setBusy(true);
    setStatus("connecting", "Connecting…");

    const src = new EventSource(`/generate?prompt=${encodeURIComponent(prompt)}`);
    activeSource = src;

    src.addEventListener("start", () => {
        typing.remove();
        setStatus("streaming", "Streaming…");
    });

    src.addEventListener("delta", (ev) => {
        typing.remove();
        assistantMsg.textContent += ev.data ?? "";
        scrollBottom();
        setStatus("streaming", "Streaming…");
    });

    src.addEventListener("sources", (ev) => {
        try {
            const chunks = JSON.parse(ev.data);
            renderEvidence(chunks);
        } catch (_) { /* ignore */ }
    });

    src.addEventListener("done", () => {
        typing.remove();
        src.close();
        activeSource = null;
        setStatus("ready", "Ready");
        setBusy(false);
    });

    src.addEventListener("error", () => {
        typing.remove();
        src.close();
        activeSource = null;
        if (!assistantMsg.textContent.trim()) {
            assistantMsg.textContent = "Connection failed. Is the server running?";
        }
        setStatus("error", "Error");
        setBusy(false);
    });
});

// ── Input shortcuts ───────────────────────────────────────────────────────────

promptInput.addEventListener("input", autoResize);
promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm.requestSubmit();
    }
});

clearBtn.addEventListener("click", () => {
    stopStream();
    chatLog.innerHTML = "";
    clearEvidence();
    setBusy(false);
    setStatus("ready", "Ready");
    promptInput.focus();
    createMessage("system", "Conversation cleared. Ask anything to start.");
});

// ── Init ──────────────────────────────────────────────────────────────────────

createMessage("system", "Ask anything to start. Click any retrieved chunk to read the full text.");
setStatus("ready", "Ready");
autoResize();
