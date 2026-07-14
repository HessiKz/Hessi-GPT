// HESSI-GPT — static inference terminal.
// The upstream endpoint + model + identity directive are stored only as
// AES-GCM ciphertext and are decrypted at runtime from a reconstructed key.
// Nothing about the backing model is readable in this file.

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

// ---------------------------------------------------------------------------
// Encrypted runtime configuration (ciphertext only).
// ---------------------------------------------------------------------------
const __CFG = {
  blob: {
    v: 2,
    kdf: { salt: "aGVzc2ktZ3B0LXN0YXRpYy1zYWx0LTIwMjY=", it: 210000 },
    iv: "wnPGL8D9KppQiGWk",
    ct: "v88roIAX+svWrwca3htRtCRm+jKPjHf4/c1748MzTer3ayJW1BG5tgT9EGqTFdpTJ9vFNFP+Qgt4qJpDt/KAt8DsGfPl1UmufS6++GJO+LqlRTq+XCKbLZt9H6psE+tpoqExlLoTYf9Mns7KfZ+WTUDNhsUgV6pvo5B218xl5KZ7e8zmNRKGpCgLS8/5inLUJqjeWbzoOhPD+vmVfqW26dGhWCx0Xkd2G7I9PPbOYOB8jl1QdwytfzVzrM90JsMbA+bU0z5jZ4FODWaSyxa3cRb2S6mXpVrnpM6FtgEXMTKbVhFahb6ecJdcfeowTVhg60GKw99IVd4M4CVskyX/Tv9AgLRwibDiFVjAyD3/XsVDVIf0HEvjx28azqIobUruvncnMrDssNUu8I4iAxfL8+WSB4DPYhBnkdNjiJiVuN38FVzO89hJv9NTIFptj68UVDwCOcBYrFDT9nUMPfsFRaKw7tOt5iuNKDuy2D9V4rH4IjOqDleMxFDnX9/G3ZMvVL1jnTXHC11VA7p6dJV+BCDCNSfCiHTFZZ0QzUzvjuryui0X6FYlXcEydiE7MrwCQe3we9AlSdZZVULSYPutt1UzvMS21puKe19/zUZ7Uf81YONtQKBW7O5e+E9djdPfEC3UBk3rvfVxqgZ11AzLbtH057NU/hhL9pEN49uXCjtIVwEF7Xhsf6bSfJ9/Kn9R5mrNuxc2Hbl54xnxa3ebLNffIv7thphcSnTRYrp/ako92b60TvcUAhrxybZcn7cwsO62rGIW343XEAODCDdduJ4ebjYn0DoSUc8sLeSx4TZCo1e88M+swbmox3mXf0PyzQL055s24Jgo317jQwSK1dT+f2mOEQzm5YmIZYqBlkfH7TCm6NpVmCBzH+Di8bHONgruyU49sa720ejXLo/bo6uA1T8O9xFjPPqZiR36bm39ThepBzZveXBT3pYGLtTrgS4hyY2flsocU/qvZBh2SDodSfcX/kR+pbcNiCAmqe15x0lJdMWyV1bVmDSnZJfXdwOFwKgiVAsAOxqwNxxqsgSFmATB1nWxAC4BSlYrtAp2QmKzlZu3CskdaOyfZnWJBoNjQCfROJeg41nd3fYk46V+7yvZfx6o19vCcEy2S2T2EmkKsE0Fs+bJDviOcOMnCG/S3kx66II86hl/yhpfon84Oj9Xrqskv+/8VCBf3hGUydc5ETMW00njybA6NQ/pHVrQB9rdFPKXig==",
  },
  xkey: [55, 90, 19, 196, 158, 107],
  shards: ["f2lgt69GcApH/qQ=", "QWg98qRR", "UzU+qvEfGj52p/EGRzN/oaRR", "ByJS9dhS"],
};

function __b642u8(s) {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function __u82str(u8) {
  let s = "";
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return s;
}

// Reassemble the passphrase from XOR-masked, base64 shards.
function __rebuildKey() {
  const xk = __CFG.xkey;
  const parts = __CFG.shards.map((b64) => {
    const bytes = __b642u8(b64);
    const clear = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) clear[i] = bytes[i] ^ xk[i % xk.length];
    return clear;
  });
  const total = parts.reduce((n, p) => n + p.length, 0);
  const merged = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    merged.set(p, off);
    off += p.length;
  }
  return __u82str(merged);
}

async function __decryptCfg() {
  const enc = new TextEncoder();
  const pass = __rebuildKey();
  const salt = __b642u8(__CFG.blob.kdf.salt);
  const baseKey = await crypto.subtle.importKey(
    "raw",
    enc.encode(pass),
    "PBKDF2",
    false,
    ["deriveKey"]
  );
  const key = await crypto.subtle.deriveKey(
    { name: "PBKDF2", salt, iterations: __CFG.blob.kdf.it, hash: "SHA-256" },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false,
    ["decrypt"]
  );
  const iv = __b642u8(__CFG.blob.iv);
  const ct = __b642u8(__CFG.blob.ct);
  const plain = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
  return JSON.parse(__u82str(new Uint8Array(plain)));
}

let CFG = null;
async function initConfig() {
  try {
    CFG = await __decryptCfg();
  } catch (e) {
    console.error("config decrypt failed", e);
    sysStatus.textContent = "FAULT";
    appendMessage("assistant", "Core configuration failed to load.");
  }
}

// ---------------------------------------------------------------------------
// Static "architecture" display (no backend on static hosting).
// ---------------------------------------------------------------------------
const HESSI_SPECS = [
  { label: "MODEL", value: "HESSI-GPT" },
  { label: "FAMILY", value: "GPT-2.5" },
  { label: "VARIANT", value: "CHAT / INSTRUCTION" },
  { label: "RUNTIME", value: "STREAMING INFERENCE" },
  { label: "SAMPLING", value: "TOP-P / TOP-K" },
  { label: "AUTHOR", value: "HESSIKZ" },
];
const HESSI_BLOCKS = [
  "Token + Position Embedding",
  "Masked Multi-Head Self-Attention",
  "LayerNorm + GELU FFN",
  "LM Head",
];
const HESSI_STEPS = [
  { id: "RESTORE_CHECKPOINT", label: "Load neural core" },
  { id: "ENCODE_PROMPT", label: "Encode operator prompt" },
  { id: "INFER", label: "Stream inference" },
  { id: "EMIT_RESPONSE", label: "Emit response" },
];

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
  if (status === "running") stateEl.textContent = "RUN…";
  else if (status === "complete") stateEl.textContent = detail?.elapsed_ms != null ? `${detail.elapsed_ms}MS` : "OK";
  else stateEl.textContent = (status || "IDLE").toUpperCase();
  setText(phaseStatus, phase.replaceAll("_", " "));
}

function appendMessage(role, body, streaming = false) {
  const article = document.createElement("article");
  article.className = `msg msg-${role}`;
  const meta = document.createElement("p");
  meta.className = "msg-meta";
  meta.textContent = role === "user" ? ">>> OPERATOR INPUT" : "<<< HESSI-GPT OUTPUT";
  const content = document.createElement("p");
  content.className = `msg-body${streaming ? " streaming" : ""}`;
  content.textContent = body;
  article.append(meta, content);
  chatLog.append(article);
  chatLog.scrollTop = chatLog.scrollHeight;
  return content;
}

// GPT-2.5-class CPU feel: real stream chunks, deliberately paced on screen.
const TOKEN_PACE_MS = 110;
const PREFILL_MS = 420;

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function extractDelta(json) {
  return (
    json?.choices?.[0]?.delta?.content ??
    json?.result?.choices?.[0]?.delta?.content ??
    ""
  );
}

function extractFullText(json) {
  return (
    json?.choices?.[0]?.message?.content ??
    json?.result?.choices?.[0]?.message?.content ??
    ""
  );
}

/** Split a finished reply into small pieces so fallback still "streams". */
function chunkTextForPace(text) {
  const pieces = [];
  const re = /(\s+|[^\s]+)/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const part = m[0];
    if (part.length <= 4) {
      pieces.push(part);
    } else {
      for (let i = 0; i < part.length; i += 2) pieces.push(part.slice(i, i + 2));
    }
  }
  return pieces.length ? pieces : [text];
}

/**
 * Live SSE reader that feeds a queue; paint loop drains it at TOKEN_PACE_MS
 * so the UI stays streaming while feeling like a small GPT-2.5 decode.
 */
function createPacedStream(assistantNode, t0) {
  const queue = [];
  let done = false;
  let full = "";
  let idx = 0;
  let wake = null;

  function notify() {
    if (wake) {
      const r = wake;
      wake = null;
      r();
    }
  }

  function push(piece) {
    if (!piece) return;
    queue.push(piece);
    notify();
  }

  function finish() {
    done = true;
    notify();
  }

  async function waitForPiece() {
    if (queue.length || done) return;
    await new Promise((r) => {
      wake = r;
    });
  }

  async function paint() {
    if (assistantNode.textContent === "Awaiting inference…") {
      assistantNode.textContent = "";
    }
    while (!done || queue.length) {
      await waitForPiece();
      while (queue.length) {
        const piece = queue.shift();
        full += piece;
        idx += 1;
        assistantNode.textContent = full;
        setText(tokenCount, pad3(idx));
        setText(telFragment, JSON.stringify(piece));
        setText(telContext, String(full.length));
        const elapsed = performance.now() - t0;
        setText(tokenLatency, `${(elapsed / Math.max(idx, 1)).toFixed(1)}MS`);
        chatLog.scrollTop = chatLog.scrollHeight;
        const jitter = 0.75 + Math.random() * 0.55;
        await sleep(Math.round(TOKEN_PACE_MS * jitter));
      }
    }
    return { full, idx };
  }

  return { push, finish, paint };
}

async function readSseIntoQueue(res, paced) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let gotAny = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      try {
        const json = JSON.parse(data);
        const delta = extractDelta(json);
        if (delta) {
          gotAny = true;
          paced.push(delta);
        }
      } catch {
        /* ignore partial frames */
      }
    }
  }
  return gotAny;
}

async function streamChat(prompt, maxTokens) {
  if (!CFG) {
    appendMessage("assistant", "Neural core unavailable.");
    return;
  }
  appendMessage("user", prompt);
  const assistantNode = appendMessage("assistant", "Awaiting inference…", true);

  setPhase("RESTORE_CHECKPOINT", "running");
  await sleep(PREFILL_MS);
  setPhase("RESTORE_CHECKPOINT", "complete");

  const t0 = performance.now();
  const messages = [
    { role: "system", content: CFG.s },
    ...conversation.map((m) => ({ role: m.role, content: m.content })),
    { role: "user", content: prompt },
  ];

  try {
    setPhase("ENCODE_PROMPT", "running");
    await sleep(180);
    setPhase("ENCODE_PROMPT", "complete");
    setPhase("INFER", "running");

    const paced = createPacedStream(assistantNode, t0);
    const paintPromise = paced.paint();

    let gotStream = false;
    const res = await fetch(`${CFG.b}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: CFG.m,
        stream: true,
        max_tokens: maxTokens,
        messages,
      }),
    });

    if (res.ok && res.body) {
      gotStream = await readSseIntoQueue(res, paced);
    }

    // Fallback: non-stream completion, still paced onto the UI.
    if (!gotStream) {
      const res2 = await fetch(`${CFG.b}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: CFG.m,
          stream: false,
          max_tokens: maxTokens,
          messages,
        }),
      });
      if (!res2.ok) {
        paced.finish();
        await paintPromise;
        assistantNode.textContent = "Transmission failed. Check connection.";
        assistantNode.classList.remove("streaming");
        setPhase("INFER", "idle");
        return;
      }
      const json = await res2.json();
      const text = extractFullText(json);
      if (!text) {
        paced.finish();
        await paintPromise;
        assistantNode.textContent = "Empty response from neural core.";
        assistantNode.classList.remove("streaming");
        setPhase("INFER", "idle");
        return;
      }
      for (const piece of chunkTextForPace(text)) paced.push(piece);
    }

    paced.finish();
    const { full, idx } = await paintPromise;

    assistantNode.classList.remove("streaming");
    setPhase("INFER", "complete");
    setPhase("EMIT_RESPONSE", "complete");
    conversation.push({ role: "user", content: prompt });
    conversation.push({ role: "assistant", content: full });
    // Keep history short so a small model stays coherent.
    if (conversation.length > 12) conversation = conversation.slice(-12);
    setText(tokenCount, pad3(idx));
    setText(phaseStatus, "IDLE");
  } catch (err) {
    assistantNode.textContent = `Error: ${err.message}`;
    assistantNode.classList.remove("streaming");
    setPhase("INFER", "idle");
  }
}

let conversation = [];
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
    promptInput.value = "";
  }
});

await initConfig();
renderSpec(HESSI_SPECS);
renderBlocks(HESSI_BLOCKS);
initPipeline(HESSI_STEPS);
setText(sysStatus, CFG ? "READY" : "FAULT");
setInterval(() => setText(sysStatus, CFG ? "READY" : "FAULT"), 15000);
