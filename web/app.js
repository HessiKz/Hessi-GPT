const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt-input");
const maxTokensInput = document.getElementById("max-tokens");
const sendBtn = document.getElementById("send-btn");
const pipelineEl = document.getElementById("pipeline");
const specList = document.getElementById("spec-list");
const blockList = document.getElementById("block-list");
const sysStatus = document.getElementById("sys-status");
const phaseStatus = document.getElementById("phase-status");
const tokenCount = document.getElementById("token-count");
const tokenLatency = document.getElementById("token-latency");
const telTokenId = document.getElementById("tel-token-id");
const telFragment = document.getElementById("tel-fragment");
const telContext = document.getElementById("tel-context");
const telPromptIds = document.getElementById("tel-prompt-ids");

const pipelineState = new Map();

function pad3(n) {
  return String(n).padStart(3, "0");
}

function setText(el, value) {
  el.textContent = value;
}

function renderSpec(items) {
  specList.replaceChildren();
  for (const row of items) {
    const wrap = document.createElement("div");
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = row.label;
    dd.textContent = row.value;
    wrap.append(dt, dd);
    specList.append(wrap);
  }
}

function renderBlocks(blocks) {
  blockList.replaceChildren();
  for (const block of blocks) {
    const li = document.createElement("li");
    li.textContent = block;
    blockList.append(li);
  }
}

function initPipeline(steps) {
  pipelineEl.replaceChildren();
  pipelineState.clear();
  steps.forEach((step, index) => {
    const li = document.createElement("li");
    li.dataset.phase = step.id;
    li.className = "idle";
    li.innerHTML = `
      <span class="idx">${pad3(index + 1)}</span>
      <span class="label">${step.label}</span>
      <span class="state">IDLE</span>
    `;
    pipelineEl.append(li);
    pipelineState.set(step.id, li);
  });
}

function setPhase(phase, status, detail) {
  const li = pipelineState.get(phase);
  if (!li) return;
  li.className = status === "running" ? "running" : status === "complete" ? "complete" : "idle";
  const stateEl = li.querySelector(".state");
  if (status === "running") {
    stateEl.textContent = "RUN…";
  } else if (status === "complete") {
    stateEl.textContent = detail?.elapsed_ms != null ? `${detail.elapsed_ms}MS` : "OK";
  } else if (status === "eos") {
    stateEl.textContent = "EOS";
  } else {
    stateEl.textContent = (status || "IDLE").toUpperCase();
  }
  setText(phaseStatus, phase.replaceAll("_", " "));
}

function appendMessage(role, body, streaming = false) {
  const article = document.createElement("article");
  article.className = `msg msg-${role}`;
  const meta = document.createElement("p");
  meta.className = "msg-meta";
  meta.textContent = role === "user" ? ">>> OPERATOR INPUT" : "<<< MODEL OUTPUT";
  const content = document.createElement("p");
  content.className = `msg-body${streaming ? " streaming" : ""}`;
  content.textContent = body;
  article.append(meta, content);
  chatLog.append(article);
  chatLog.scrollTop = chatLog.scrollHeight;
  return content;
}

async function loadArchitecture() {
  const res = await fetch("/api/architecture");
  const data = await res.json();
  initPipeline(data.steps || []);
  renderSpec(data.spec || []);
  renderBlocks(data.blocks || []);
}

async function loadHealth() {
  const res = await fetch("/api/health");
  const data = await res.json();
  setText(sysStatus, data.status === "ready" ? "READY" : "BOOTING…");
}

async function streamChat(prompt, maxTokens) {
  appendMessage("user", prompt);
  const assistantNode = appendMessage("assistant", "Awaiting inference…", true);

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, max_tokens: maxTokens }),
  });

  if (!res.ok || !res.body) {
    assistantNode.textContent = "Transmission failed. Check server logs.";
    assistantNode.classList.remove("streaming");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";

    for (const chunk of chunks) {
      const line = chunk.trim();
      if (!line.startsWith("data:")) continue;
      const payload = JSON.parse(line.slice(5).trim());
      handleEvent(payload, assistantNode);
    }
  }
}

function handleEvent(event, assistantNode) {
  if (event.type === "step") {
    setPhase(event.phase, event.status, event);
    if (event.phase === "ENCODE_PROMPT" && event.status === "complete") {
      setText(telPromptIds, (event.token_ids || []).join(", "));
    }
    if (event.phase === "RESTORE_CHECKPOINT" && event.status === "complete" && event.params) {
      renderSpec([
        { label: "PARAM COUNT", value: String(event.params) },
        ...Array.from(specList.querySelectorAll("div"))
          .slice(1)
          .map((div) => ({
            label: div.querySelector("dt").textContent,
            value: div.querySelector("dd").textContent,
          })),
      ]);
    }
    return;
  }

  if (event.type === "token") {
    assistantNode.textContent = event.partial || assistantNode.textContent;
    setText(tokenCount, pad3(event.index || 0));
    setText(tokenLatency, event.latency_ms != null ? `${event.latency_ms.toFixed(1)}MS` : "—");
    setText(telTokenId, String(event.token_id ?? "—"));
    setText(telFragment, JSON.stringify(event.token_text ?? ""));
    setText(telContext, String(event.context_len ?? "—"));
    if (event.status === "eos") {
      assistantNode.classList.remove("streaming");
    }
    return;
  }

  if (event.type === "done") {
    assistantNode.textContent = event.text;
    assistantNode.classList.remove("streaming");
    setPhase("EMIT_RESPONSE", "complete", event);
    setText(tokenCount, pad3(event.tokens_generated || 0));
    setText(phaseStatus, "IDLE");
    return;
  }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  sendBtn.disabled = true;
  sendBtn.textContent = "Transmitting…";

  for (const li of pipelineState.values()) {
    li.className = "idle";
    li.querySelector(".state").textContent = "IDLE";
  }

  try {
    await streamChat(prompt, Number(maxTokensInput.value) || 120);
  } catch (err) {
    appendMessage("assistant", `Error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Transmit Prompt";
  }
});

await loadArchitecture();
await loadHealth();
setInterval(loadHealth, 15000);
