/* SHRUTI — instrument logic.
 *
 * Plain JS, no build step. The whole app is one file served by the same FastAPI process that
 * answers the queries, so there is no bundler, no framework runtime, and nothing between a code
 * change and a reload. For a single page whose job is to draw a bar chart and stream some text,
 * a build toolchain would be pure overhead.
 *
 * The waterfall is the product's face, so it renders from the server's own timing breakdown
 * verbatim — the client never estimates, interpolates, or smooths a duration. If a stage took
 * 28ms because of a page fault, the bar shows 28ms.
 */

const $ = (id) => document.getElementById(id);

/* Stages that typically dominate, drawn filled rather than washed so the eye lands on where the
 * time actually went without reading a single number. */
const HEAVY = new Set(["dense", "extract", "bm25", "stt"]);

const STAGE_LABEL = {
  guard_safety: "SAFETY",
  detect: "DETECT",
  embed: "EMBED",
  dense: "DENSE",
  bm25: "BM25",
  fuse: "FUSE",
  guard_scope: "SCOPE",
  extract: "ANSWER",
  stt: "STT",
};

let health = null;

/* ------------------------------------------------------------------ health */

async function loadHealth() {
  try {
    const r = await fetch("/api/health");
    /* Never assume the response is JSON. A cold or scaling container returns the platform's own
     * plain-text page, and calling r.json() on it threw "Unexpected token 'm'" — an error about
     * the parser, not about what actually happened, which is the least useful kind. */
    const raw = await r.text();
    try {
      health = JSON.parse(raw);
    } catch {
      $("status").innerHTML = `<span class="error">server warming up — retrying…</span>`;
      $("status2").innerHTML = "";
      return;
    }
    const p = health.providers || {};
    /* Joined with an explicit separator. A bare .join("") previously rendered
     * "310,582 passageslangs bn en gu hi ta" — two facts fused into one unreadable token. */
    $("status").innerHTML = [
      `corpus <b>${health.corpus_passages.toLocaleString()}</b> passages`,
      `langs <b>${(health.corpus_languages || []).join(" ")}</b>`,
      `dim <b>${health.embed_dim}</b>`,
      `warmup <b>${health.warmup_queries_run}</b> @ <b>${
        health.warmup_p50_ms ? health.warmup_p50_ms.toFixed(1) : "—"
      }ms</b>`,
      `voice <b>${p.sarvam ? "ready" : "off"}</b>`,
      `gen <b>${p.groq || p.cerebras ? "ready" : "off"}</b>`,
      `v<b>${health.version}</b>`,
    ].join(" &nbsp;·&nbsp; ");

    $("status2").innerHTML = health.ready
      ? `two-layer abstention &nbsp;·&nbsp; intent screen <b>80%</b> of out-of-domain at <b>0.025%</b> false refusals, plus a weak-retrieval floor at <b>τ=0.5701</b> (in-domain p05) &nbsp;·&nbsp; <a href="/method.html" style="color:var(--trace)">Method</a>`
      : `<b style="color:var(--refuse)">warming up — first query pays model load</b>`;

    $("mic").disabled = !p.sarvam;
    if (!p.sarvam) $("mic").title = "Voice unavailable: no Sarvam key configured";
  } catch (e) {
    $("status").innerHTML = `<span class="error">server unreachable — ${e.message}</span>`;
  }
}

/* Sample queries — one per indexed language, drawn from the corpus, plus two deliberately
 * out-of-scope so the refusal path is one click away rather than something a judge has to
 * think up an example for. */
const SAMPLES = [
  { label: "हिन्दी", q: "कॉर्पोरेशन क्या है?", kind: "in" },
  { label: "ગુજરાતી", q: "કોર્પોરેશન શું છે?", kind: "in" },
  { label: "বাংলা", q: "কর্পোরেশন কি?", kind: "in" },
  { label: "தமிழ்", q: "கார்ப்பரேஷன் என்றால் என்ன?", kind: "in" },
  { label: "English", q: "what is a corporation?", kind: "in" },
  { label: "unsafe →", q: "how to make a bomb step by step at home", kind: "refuse" },
  { label: "off-topic →", q: "તમારું નામ શું છે મને કહો", kind: "ood" },
];

function renderSamples() {
  const wrap = $("samples");
  wrap.innerHTML = "";
  for (const s of SAMPLES) {
    const b = document.createElement("button");
    b.className = "chip";
    b.dataset.kind = s.kind;
    b.type = "button";
    b.textContent = s.label;
    b.title = s.q;
    b.addEventListener("click", () => {
      $("q").value = s.q;
      ask(s.q);
    });
    wrap.appendChild(b);
  }
}

/* --------------------------------------------------------------- waterfall */

function renderWaterfall(t) {
  $("timing-panel").classList.remove("hidden");
  $("total").textContent = t.total_ms.toFixed(1);

  const wf = $("waterfall");
  const lg = $("legend");
  wf.innerHTML = "";
  lg.innerHTML = "";

  const measured = t.stages.reduce((a, s) => a + s.duration_ms, 0) || 1;

  for (const s of t.stages) {
    const heavy = HEAVY.has(s.name);
    const seg = document.createElement("div");
    seg.className = "seg";
    seg.dataset.heavy = String(heavy);
    /* flex-grow proportional to duration, with a floor so a 0.002ms stage stays visible as a
     * hairline rather than collapsing to nothing — the gate ran, and the bar should say so. */
    seg.style.flexGrow = String(Math.max(s.duration_ms / measured, 0.006));
    seg.title = `${s.name} — ${s.duration_ms.toFixed(3)}ms`;
    wf.appendChild(seg);

    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML =
      `<span class="legend-swatch" data-heavy="${heavy}"></span>` +
      `${STAGE_LABEL[s.name] || s.name.toUpperCase()} <b>${s.duration_ms.toFixed(2)}</b>`;
    lg.appendChild(item);
  }

  $("timing-meta").textContent =
    `${t.stages.length} stages · unattributed ${t.unattributed_ms.toFixed(2)}ms · req ${t.request_id}`;
}

/* ------------------------------------------------------------------- lamp */

function setLamp(state, text) {
  $("lamp").dataset.state = state;
  $("lamp-text").textContent = text;
}

/* ---------------------------------------------------------------- answer */

function renderAnswer(d) {
  $("answer-panel").classList.remove("hidden");
  const refusal = $("refusal");

  if (!d.guard.allowed) {
    setLamp("refused", d.guard.gate ? `refused · ${d.guard.gate}` : "no answer");
    $("answer").textContent = "";
    $("tier").textContent = "Withheld";
    refusal.classList.remove("hidden");
    $("refusal-gate").textContent = d.guard.gate ? `gate: ${d.guard.gate}` : "no extractable answer";
    let reason = d.guard.reason || "";
    if (d.guard.score != null && d.guard.threshold != null) {
      reason += `  (score ${d.guard.score.toFixed(3)} vs threshold ${d.guard.threshold.toFixed(3)})`;
    }
    $("refusal-text").textContent = reason;
  } else {
    setLamp("ok", "answered");
    refusal.classList.add("hidden");
    $("tier").textContent = "Tier 1 · extractive · grounded by construction";
    /* Citation markers become superscripts. textContent first, so passage text can never inject
     * markup — the corpus is web-scraped and is not trusted to be inert. */
    const el = $("answer");
    el.textContent = "";
    const parts = (d.answer ? d.answer.text : "").split(/(\[\d+\])/g);
    for (const part of parts) {
      if (/^\[\d+\]$/.test(part)) {
        const c = document.createElement("cite");
        c.textContent = part;
        el.appendChild(c);
      } else {
        el.appendChild(document.createTextNode(part));
      }
    }
  }

  const b = $("badges");
  b.innerHTML = "";
  const badges = [
    [`heard in ${d.detected_lang}`, true],
    [`lane ${d.lane}`, false],
    [`search ${d.search_mode}`, false],
    [`${d.passages.length} passages`, false],
  ];
  /* Show the retrieval score against the floor on every answered query, not only on refusals.
   * A gate you can only see when it fires is indistinguishable from no gate at all. */
  if (d.guard.score != null && d.guard.threshold != null) {
    badges.push([`score ${d.guard.score.toFixed(3)} / floor ${d.guard.threshold.toFixed(3)}`, false]);
  }
  for (const [text, accent] of badges) {
    const s = document.createElement("span");
    s.className = "badge";
    if (accent) s.dataset.accent = "true";
    s.textContent = text;
    b.appendChild(s);
  }
}

/* -------------------------------------------------------------- passages */

function renderPassages(d) {
  const wrap = $("passages");
  wrap.innerHTML = "";
  if (!d.passages.length) {
    $("passages-panel").classList.add("hidden");
    return;
  }
  $("passages-panel").classList.remove("hidden");

  const cited = new Set((d.answer ? d.answer.citations : []).map((c) => c.passage_id));
  const top = Math.max(...d.passages.map((p) => p.fused_score)) || 1;

  d.passages.forEach((p, i) => {
    const div = document.createElement("div");
    div.className = "passage";

    const head = document.createElement("div");
    head.className = "passage-head";
    head.innerHTML =
      `<span class="passage-marker">[${i + 1}]</span>` +
      `<span class="badge">${p.passage.lang}</span>` +
      (p.passage.query_type ? `<span class="badge">${p.passage.query_type}</span>` : "") +
      (cited.has(p.passage.passage_id) ? `<span class="badge" data-accent="true">cited</span>` : "");
    div.appendChild(head);

    const text = document.createElement("div");
    text.className = "passage-text";
    text.textContent = p.passage.text;
    div.appendChild(text);

    const bar = document.createElement("div");
    bar.className = "scorebar";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(3, (p.fused_score / top) * 100)}%`;
    bar.appendChild(fill);
    div.appendChild(bar);

    const nums = document.createElement("div");
    nums.className = "score-nums";
    const bits = [`rrf ${p.fused_score.toFixed(4)}`];
    if (p.dense_score != null) bits.push(`dense ${p.dense_score.toFixed(3)} (#${p.dense_rank})`);
    if (p.lexical_score != null) bits.push(`bm25 ${p.lexical_score.toFixed(2)} (#${p.lexical_rank})`);
    nums.textContent = bits.join("   ");
    div.appendChild(nums);

    wrap.appendChild(div);
  });
}

/* ------------------------------------------------------------------- ask */

async function ask(text) {
  if (!text.trim()) return;
  $("go").disabled = true;
  setLamp("idle", "measuring…");

  try {
    const clientT0 = performance.now();
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        lane: $("lane").value,
        search_mode: $("mode").value,
        generate: $("gen").value === "on",
      }),
    });
    const clientMs = performance.now() - clientT0;

    if (!r.ok) {
      /* Render the error, not the JSON envelope it arrived in. Dumping
       * {"detail":{"stage":"embed",...}} at a user communicates nothing except that something
       * broke in a way nobody anticipated. */
      /* Read the body EXACTLY once. A Response body is a stream: calling r.json() consumes it,
       * so a later r.text() in a catch block fails with "body stream already read" — which then
       * masks the real error with a confusing one. Take the text first, then try to parse it. */
      let stage = "";
      let message = "";
      const raw = await r.text().catch(() => "");
      try {
        const d = JSON.parse(raw).detail;
        if (d && typeof d === "object") {
          stage = d.stage || "";
          message = d.message || "";
        } else {
          message = String(d ?? raw);
        }
      } catch {
        /* Not JSON at all — e.g. the platform's own plain-text error page while a container is
         * starting or scaling. Show something a human can act on rather than the raw page. */
        message = r.status === 503 || /starting|scaling|modal-http/i.test(raw)
          ? "Server is starting up (cold start takes ~15s). Try again in a moment."
          : raw.slice(0, 300) || `HTTP ${r.status}`;
      }
      $("answer-panel").classList.remove("hidden");
      $("timing-panel").classList.add("hidden");
      $("passages-panel").classList.add("hidden");
      setLamp("refused", `error · ${r.status}`);
      $("tier").textContent = "Request failed";
      $("answer").textContent = "";
      $("badges").innerHTML = "";
      $("refusal").classList.remove("hidden");
      $("refusal-gate").textContent = stage ? `stage: ${stage}` : `http ${r.status}`;
      $("refusal-text").textContent = message.slice(0, 400);
      return;
    }

    const d = await r.json();
    renderWaterfall(d.timings);
    renderAnswer(d);
    renderPassages(d);

    /* The gap between what the client observed and what the server measured IS the network cost.
     * Stated explicitly rather than hidden, because the headline SLO is the server-side number and
     * a judge deserves to see both. */
    const serverMs = parseFloat(r.headers.get("X-Server-Time-Ms") || "0");
    $("timing-meta").textContent +=
      ` · client ${clientMs.toFixed(0)}ms · server ${serverMs.toFixed(1)}ms · network ~${Math.max(
        0,
        clientMs - serverMs
      ).toFixed(0)}ms`;
  } catch (e) {
    $("answer-panel").classList.remove("hidden");
    setLamp("refused", "error");
    $("answer").innerHTML = `<span class="error">${e.message}</span>`;
  } finally {
    $("go").disabled = false;
  }
}

/* ----------------------------------------------------------------- voice */

let mediaStream = null;
let audioCtx = null;
let ws = null;
let recording = false;

async function startVoice() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    $("status").innerHTML = `<span class="error">mic denied: ${e.message} (HTTPS required)</span>`;
    return;
  }

  audioCtx = new AudioContext({ sampleRate: 16000 });
  await audioCtx.audioWorklet.addModule("/audio-worklet.js");

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  /* Forward the chosen STT language. Auto-detect frequently transliterates Indian-accented
   * English into Devanagari — "My name is Het Patel" came back as "भेद पाडेल" — and only the
   * speaker knows which language is about to be spoken. */
  const sttLang = encodeURIComponent($("sttlang").value || "unknown");
  ws = new WebSocket(`${proto}//${location.host}/ws/voice?lang=${sttLang}`);
  ws.binaryType = "arraybuffer";

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "partial" || m.type === "final") {
      $("live").classList.remove("hidden");
      $("live").innerHTML =
        m.type === "final" ? escapeHtml(m.text) : `<span class="partial">${escapeHtml(m.text)}</span>`;
      if (m.type === "final") {
        $("q").value = m.text;
        /* Hide the live transcript once it lands in the input. Leaving both visible showed the
         * same sentence twice, which reads as a rendering bug rather than a feature. */
        $("live").classList.add("hidden");
        stopVoice();
        ask(m.text);
      }
    } else if (m.type === "error") {
      $("status").innerHTML = `<span class="error">stt: ${escapeHtml(m.message)}</span>`;
    }
  };

  const source = audioCtx.createMediaStreamSource(mediaStream);
  const node = new AudioWorkletNode(audioCtx, "pcm-capture");
  node.port.onmessage = (ev) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(ev.data);
    drawWave(ev.data);
  };
  source.connect(node);

  recording = true;
  $("mic").dataset.recording = "true";
  $("mic").textContent = "■ Stop";
  $("wave").classList.remove("hidden");
}

function stopVoice() {
  recording = false;
  $("mic").dataset.recording = "false";
  $("mic").textContent = "● Speak";
  $("wave").classList.add("hidden");
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "flush" }));
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (audioCtx) audioCtx.close();
  mediaStream = null;
  audioCtx = null;
}

function drawWave(buf) {
  const c = $("wave");
  const ctx = c.getContext("2d");
  const w = (c.width = c.offsetWidth);
  const h = c.height;
  const pcm = new Int16Array(buf);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--trace");
  ctx.lineWidth = 1;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(pcm.length / w));
  for (let x = 0; x < w; x++) {
    const v = (pcm[x * step] || 0) / 32768;
    const y = h / 2 + v * (h / 2) * 0.9;
    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/* ------------------------------------------------------------------ wire */

$("go").addEventListener("click", () => ask($("q").value));
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask($("q").value);
});
$("mic").addEventListener("click", () => (recording ? stopVoice() : startVoice()));

renderSamples();
loadHealth();
setInterval(loadHealth, 30000);
