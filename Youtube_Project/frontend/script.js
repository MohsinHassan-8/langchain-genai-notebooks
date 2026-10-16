const chatWindow = document.getElementById("chatWindow");
const emptyState = document.getElementById("emptyState");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const videoIdInput = document.getElementById("videoId");
const kSlider = document.getElementById("kSlider");
const kValue = document.getElementById("kValue");
const loadVideoBtn = document.getElementById("loadVideoBtn");
const clearChatBtn = document.getElementById("clearChatBtn");
const statusPill = document.getElementById("statusPill");

let messages = []; // { role: 'user'|'assistant', content, sources? }

kSlider.addEventListener("input", () => {
  kValue.textContent = kSlider.value;
});

function setStatus(text, kind) {
  statusPill.textContent = text;
  statusPill.className = `status-pill ${kind}`;
}

function renderMessages() {
  chatWindow.innerHTML = "";

  if (messages.length === 0) {
    chatWindow.appendChild(emptyState);
    return;
  }

  messages.forEach((msg, idx) => {
    const row = document.createElement("div");
    row.className = `message-row ${msg.role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = msg.role === "user" ? "🧑‍💻" : "🎬";

    const wrap = document.createElement("div");
    wrap.className = "bubble-wrap";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = msg.content;
    wrap.appendChild(bubble);

    if (msg.sources && msg.sources.length > 0) {
      const toggle = document.createElement("button");
      toggle.className = "sources-toggle";
      toggle.textContent = `📄 View ${msg.sources.length} source excerpt(s)`;

      const panel = document.createElement("div");
      panel.className = "sources-panel";
      msg.sources.forEach((src, i) => {
        const chunk = document.createElement("div");
        chunk.className = "source-chunk";
        chunk.innerHTML = `<b>Source ${i + 1}</b><br>${escapeHtml(src.content)}`;
        panel.appendChild(chunk);
      });

      toggle.addEventListener("click", () => {
        panel.classList.toggle("open");
      });

      wrap.appendChild(toggle);
      wrap.appendChild(panel);
    }

    row.appendChild(avatar);
    row.appendChild(wrap);
    chatWindow.appendChild(row);
  });

  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function showTypingIndicator() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.id = "typingRow";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "🎬";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatWindow.appendChild(row);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeTypingIndicator() {
  const row = document.getElementById("typingRow");
  if (row) row.remove();
}

async function askQuestion(question) {
  const videoId = videoIdInput.value.trim();
  const k = parseInt(kSlider.value, 10);

  messages.push({ role: "user", content: question });
  renderMessages();
  showTypingIndicator();

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, video_id: videoId, k }),
    });

    removeTypingIndicator();

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Something went wrong." }));
      messages.push({ role: "assistant", content: `⚠️ ${err.detail}` });
      setStatus("Error", "error");
    } else {
      const data = await res.json();
      messages.push({
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      });
      setStatus("Ready", "ready");
    }
  } catch (e) {
    removeTypingIndicator();
    messages.push({ role: "assistant", content: "⚠️ Could not reach the server." });
    setStatus("Error", "error");
  }

  renderMessages();
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  askQuestion(question);
});

clearChatBtn.addEventListener("click", () => {
  messages = [];
  renderMessages();
});

loadVideoBtn.addEventListener("click", async () => {
  const videoId = videoIdInput.value.trim();
  const k = parseInt(kSlider.value, 10);
  if (!videoId) return;

  setStatus("Loading...", "loading");
  messages = [];
  renderMessages();

  try {
    // Force rebuild for this video/k, then do a lightweight "warm-up" ask
    await fetch(`/api/reset?video_id=${encodeURIComponent(videoId)}&k=${k}`, {
      method: "POST",
    });
    setStatus("Ready", "ready");
    messages.push({
      role: "assistant",
      content: `Loaded video "${videoId}". Ask me anything about it!`,
    });
    renderMessages();
  } catch (e) {
    setStatus("Error", "error");
  }
});

renderMessages();
