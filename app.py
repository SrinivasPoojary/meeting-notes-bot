import os
import sqlite3
import datetime
import time

import requests
import streamlit as st

DB_FILE = "meetings.db"
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL = "whisper-large-v3"
MAX_MB = 25  # Groq free tier upload limit

NOTES_SYSTEM = """You write accurate meeting notes from a transcript.
Rules:
- Use ONLY what is in the transcript. Never invent facts, names, numbers or dates.
- If something is unclear or cut off, write [unclear] instead of guessing.
- Keep names, numbers and dates exactly as spoken.
- Write in simple, clear English.

Use exactly this format:

## Summary
(3-5 sentences)

## Key points
- ...

## Decisions made
- ... (write "None mentioned" if there are none)

## Action items
- [Owner] Task (due date if mentioned)  (write "None mentioned" if there are none)

## Open questions
- ... (write "None mentioned" if there are none)
"""


def get_secret(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


# ---------- database ----------
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created TEXT,
            title TEXT,
            transcript TEXT,
            notes TEXT
        )"""
    )
    return conn


def save_meeting(title, transcript, notes):
    conn = db()
    conn.execute(
        "INSERT INTO meetings (created, title, transcript, notes) VALUES (?, ?, ?, ?)",
        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), title, transcript, notes),
    )
    conn.commit()
    conn.close()


def load_meetings():
    conn = db()
    rows = conn.execute(
        "SELECT id, created, title, transcript, notes FROM meetings ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def delete_meeting(meeting_id):
    conn = db()
    conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    conn.commit()
    conn.close()


# ---------- speech to text ----------
def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def transcribe(audio_bytes, filename, language, vocab):
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it in secrets.")
    data = {
        "model": WHISPER_MODEL,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if language != "Auto-detect":
        data["language"] = {"English": "en", "Hindi": "hi", "Marathi": "mr"}[language]
    if vocab.strip():
        # Hints help Whisper spell names and terms correctly
        data["prompt"] = vocab.strip()[:800]
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}"},
        files={"file": (filename, audio_bytes)},
        data=data,
        timeout=600,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Transcription failed ({resp.status_code}): {resp.text[:300]}")
    result = resp.json()
    segments = result.get("segments") or []
    if segments:
        lines = [f"[{fmt_time(s['start'])}] {s['text'].strip()}" for s in segments]
        return "\n".join(lines)
    return result.get("text", "").strip()


# ---------- notes ----------
PREFERRED_NOTES_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
NOT_CHAT_WORDS = ("whisper", "guard", "tts", "orpheus", "playai", "embed", "safeguard", "distil")


@st.cache_data(ttl=3600, show_spinner=False)
def groq_available_models(key):
    """Ask Groq which models this account can use right now."""
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return []


def notes_model_candidates(key):
    override = get_secret("GROQ_NOTES_MODEL")
    available = groq_available_models(key)
    candidates = [override] if override else []
    # Our preferred models first, only if Groq says they exist
    candidates += [m for m in PREFERRED_NOTES_MODELS if m in available]
    # Then any other chat model on the account
    candidates += [
        m for m in available
        if m not in candidates and not any(w in m.lower() for w in NOT_CHAT_WORDS)
    ]
    # If the model list could not be fetched, still try the preferred names
    if not available:
        candidates += PREFERRED_NOTES_MODELS
    return candidates[:6]


PART_SYSTEM = NOTES_SYSTEM + """
This is only part {i} of {n} of a longer recording. Write notes for this part only."""

MERGE_SYSTEM = """You combine meeting notes written for consecutive parts of one long recording into a single set of notes.
Rules:
- Use ONLY the notes provided. Never add facts, names, numbers or dates.
- Keep every decision, action item (with owner and due date if given) and open question. Remove duplicates.
- Write one Summary of the whole recording in 4-6 sentences.
- Use exactly the same format as the notes you are given: Summary, Key points, Decisions made, Action items, Open questions.
- Write in simple, clear English."""

PAUSE_SECONDS = 12         # short wait between requests to respect the free token limit
MERGE_LIMIT_CHARS = 9000   # how much part-notes text we merge in one request


def retry_wait(resp):
    try:
        seconds = float(resp.headers.get("retry-after", "") or 0)
    except ValueError:
        seconds = 0
    return min(max(seconds, 8) + 1, 65)


def groq_chat(system, user, max_tokens=2500):
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it in secrets.")
    last_error = "No notes model is available on this Groq account."
    for model in notes_model_candidates(key):
        for attempt in range(4):
            payload = {
                "model": model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if model.startswith("openai/gpt-oss"):
                payload["reasoning_effort"] = "low"
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=300,
            )
            if resp.status_code == 200:
                text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
                if text:
                    return text
                last_error = f"{model} returned empty notes."
                break
            last_error = f"{model}: {resp.status_code} {resp.text[:200]}"
            if resp.status_code == 401:
                raise RuntimeError("Your Groq key was not accepted (401). Check GROQ_API_KEY in Secrets.")
            if resp.status_code == 429 and attempt < 3:
                time.sleep(retry_wait(resp))  # rate limit: wait, then try the same model again
                continue
            break  # otherwise move on to the next model
    raise RuntimeError(f"Notes failed. Last error: {last_error}")


def split_transcript(transcript):
    """Cut a long transcript into parts small enough for the free notes model."""
    non_ascii = sum(1 for c in transcript if ord(c) > 127)
    limit = 3000 if non_ascii / max(len(transcript), 1) > 0.3 else 8000
    parts, current, size = [], [], 0
    for line in transcript.splitlines():
        pieces = [line[i:i + limit] for i in range(0, len(line), limit)] or [""]
        for piece in pieces:
            if current and size + len(piece) + 1 > limit:
                parts.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
    if current:
        parts.append("\n".join(current))
    return parts


def merge_notes(notes_list, ctx, say):
    level = list(notes_list)
    while len(level) > 1:
        total = sum(len(n) for n in level)
        group_size = len(level) if total <= MERGE_LIMIT_CHARS else 3
        next_level = []
        for start in range(0, len(level), group_size):
            group = level[start:start + group_size]
            if len(group) == 1:
                next_level.append(group[0])
                continue
            say("Combining the parts into final notes...")
            time.sleep(PAUSE_SECONDS)
            joined = "\n\n".join(f"### Notes for part {k}\n{t}" for k, t in enumerate(group, 1))
            next_level.append(groq_chat(MERGE_SYSTEM, ctx + joined, 2500))
        level = next_level
    return level[0]


def make_notes(transcript, context, progress=None):
    say = progress or (lambda message: None)
    ctx = f"Meeting context from the user:\n{context.strip()}\n\n" if context.strip() else ""

    key = get_secret("ANTHROPIC_API_KEY")
    if key:
        import anthropic

        model = get_secret("CLAUDE_MODEL") or "claude-sonnet-5-5"
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model,
            max_tokens=3000,
            temperature=0,
            system=NOTES_SYSTEM,
            messages=[{"role": "user", "content": ctx + f"Transcript:\n{transcript}"}],
        )
        return msg.content[0].text

    # Free option: Groq writes the notes, in parts when the recording is long
    parts = split_transcript(transcript)
    if len(parts) == 1:
        return groq_chat(NOTES_SYSTEM, ctx + f"Transcript:\n{transcript}", 2500)

    partial = []
    for i, part in enumerate(parts, 1):
        say(f"Writing notes for part {i} of {len(parts)}...")
        if i > 1:
            time.sleep(PAUSE_SECONDS)
        partial.append(
            groq_chat(
                PART_SYSTEM.format(i=i, n=len(parts)),
                ctx + f"Transcript of part {i} of {len(parts)}:\n{part}",
                1800,
            )
        )
    return merge_notes(partial, ctx, say)


# ---------- screen ----------
import re

APP_NAME = "AI Voice Recording"

import base64
import pathlib

import streamlit.components.v1 as components

RECORDER_HTML = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --fg: #111827; --danger: #dc2626; }
  html, body { margin: 0; padding: 0; background: transparent; color: var(--fg);
    font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 4px 2px 8px; }
  .time { font-size: 38px; font-weight: 700; font-variant-numeric: tabular-nums; letter-spacing: 1px; line-height: 1.1; }
  .status { display: flex; align-items: center; gap: 8px; font-size: 13px; opacity: 0.8; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #9ca3af; }
  .dot.rec { background: var(--danger); animation: pulse 1.2s infinite; }
  .dot.pause { background: #f59e0b; }
  .dot.done { background: #16a34a; }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.55); }
    70% { box-shadow: 0 0 0 9px rgba(220, 38, 38, 0); }
    100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
  }
  .btns { display: flex; gap: 10px; width: 100%; }
  button { flex: 1; display: flex; align-items: center; justify-content: center; gap: 8px; border: none;
    border-radius: 12px; padding: 13px 10px; font-size: 15px; font-weight: 600; cursor: pointer;
    color: #ffffff; font-family: inherit; transition: transform 0.12s ease, opacity 0.12s ease; }
  button:hover:not(:disabled) { transform: translateY(-1px); }
  #main { background: linear-gradient(135deg, #2563eb, #4f46e5); flex: 2; }
  #stop { background: #dc2626; }
  button:disabled { opacity: 0.35; cursor: not-allowed; }
  svg { width: 18px; height: 18px; fill: currentColor; }
  audio { width: 100%; height: 38px; display: none; }
  .err { color: var(--danger); font-size: 13px; text-align: center; min-height: 0; }
</style>
</head>
<body>
<div class="wrap">
  <div class="time" id="time">00:00</div>
  <div class="status"><span class="dot" id="dot"></span><span id="status">Ready</span></div>
  <div class="btns">
    <button id="main"></button>
    <button id="stop"></button>
  </div>
  <audio id="player" controls></audio>
  <div class="err" id="err"></div>
</div>
<script>
  var ICON_REC = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="7"></circle></svg>';
  var ICON_PAUSE = '<svg viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" rx="1"></rect><rect x="14" y="5" width="4" height="14" rx="1"></rect></svg>';
  var ICON_PLAY = '<svg viewBox="0 0 24 24"><polygon points="7,4 20,12 7,20"></polygon></svg>';
  var ICON_STOP = '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"></rect></svg>';

  function post(type, data) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: type }, data || {}), "*");
  }
  function setHeight() {
    post("streamlit:setFrameHeight", { height: document.documentElement.scrollHeight });
  }
  function sendValue(value) {
    post("streamlit:setComponentValue", { value: value, dataType: "json" });
  }
  function $(id) { return document.getElementById(id); }

  var state = "idle";   // idle, recording, paused, done
  var recorder = null, stream = null, chunks = [], mime = "";
  var startedAt = 0, acc = 0, timer = null, lastUrl = null;

  function pickMime() {
    var options = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
    if (!window.MediaRecorder) return "";
    for (var i = 0; i < options.length; i++) {
      if (MediaRecorder.isTypeSupported(options[i])) return options[i];
    }
    return "";
  }
  function extFor(type) {
    if (type.indexOf("mp4") !== -1) return "mp4";
    if (type.indexOf("ogg") !== -1) return "ogg";
    return "webm";
  }
  function fmt(ms) {
    var s = Math.floor(ms / 1000);
    var m = Math.floor(s / 60);
    var r = s % 60;
    return (m < 10 ? "0" : "") + m + ":" + (r < 10 ? "0" : "") + r;
  }
  function elapsed() { return acc + (state === "recording" ? Date.now() - startedAt : 0); }
  function tick() { $("time").textContent = fmt(elapsed()); }

  function render() {
    var main = $("main"), stop = $("stop"), dot = $("dot"), status = $("status");
    stop.innerHTML = ICON_STOP + "<span>Stop</span>";
    dot.className = "dot";
    if (state === "idle") {
      main.innerHTML = ICON_REC + "<span>Record</span>";
      stop.disabled = true; status.textContent = "Ready";
    } else if (state === "recording") {
      main.innerHTML = ICON_PAUSE + "<span>Pause</span>";
      stop.disabled = false; status.textContent = "Recording"; dot.className = "dot rec";
    } else if (state === "paused") {
      main.innerHTML = ICON_PLAY + "<span>Resume</span>";
      stop.disabled = false; status.textContent = "Paused"; dot.className = "dot pause";
    } else {
      main.innerHTML = ICON_REC + "<span>Record again</span>";
      stop.disabled = true; status.textContent = "Recording saved"; dot.className = "dot done";
    }
    tick();
    setHeight();
  }

  async function start() {
    $("err").textContent = "";
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      $("err").textContent = "This browser cannot record audio. Please use Chrome.";
      setHeight();
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch (e) {
      $("err").textContent = "Microphone access is blocked. Allow the microphone in your browser settings and try again.";
      setHeight();
      return;
    }
    mime = pickMime();
    chunks = [];
    acc = 0;
    var opts = { audioBitsPerSecond: 32000 };
    if (mime) { opts.mimeType = mime; }
    try { recorder = new MediaRecorder(stream, opts); } catch (e) { recorder = new MediaRecorder(stream); }
    recorder.ondataavailable = function (e) { if (e.data && e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = finish;
    recorder.start(1000);
    startedAt = Date.now();
    state = "recording";
    clearInterval(timer);
    timer = setInterval(tick, 250);
    $("player").style.display = "none";
    sendValue(null);
    render();
  }
  function pause() {
    recorder.pause();
    acc += Date.now() - startedAt;
    state = "paused";
    render();
  }
  function resume() {
    recorder.resume();
    startedAt = Date.now();
    state = "recording";
    render();
  }
  function stopRec() {
    if (state === "recording") { acc += Date.now() - startedAt; }
    state = "done";
    clearInterval(timer);
    try { recorder.stop(); } catch (e) {}
    if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); }
    render();
  }
  function finish() {
    var fullType = recorder.mimeType || mime || "audio/webm";
    var blob = new Blob(chunks, { type: fullType });
    if (lastUrl) { URL.revokeObjectURL(lastUrl); }
    lastUrl = URL.createObjectURL(blob);
    var player = $("player");
    player.src = lastUrl;
    player.style.display = "block";
    setHeight();
    var reader = new FileReader();
    reader.onloadend = function () {
      var b64 = String(reader.result).split(",")[1] || "";
      sendValue({ audio: b64, ext: extFor(fullType), mime: fullType.split(";")[0], seconds: Math.round(acc / 1000), id: Date.now() });
    };
    reader.readAsDataURL(blob);
  }

  $("main").onclick = function () {
    if (state === "idle" || state === "done") start();
    else if (state === "recording") pause();
    else if (state === "paused") resume();
  };
  $("stop").onclick = stopRec;

  window.addEventListener("message", function (event) {
    if (event.data && event.data.type === "streamlit:render") {
      var theme = event.data.theme;
      if (theme && theme.textColor) {
        document.documentElement.style.setProperty("--fg", theme.textColor);
      }
      setHeight();
    }
  });

  render();
  post("streamlit:componentReady", { apiVersion: 1 });
  setHeight();
</script>
</body>
</html>
'''

RECORDER_DIR = pathlib.Path(__file__).parent / "voice_recorder_component"
RECORDER_DIR.mkdir(exist_ok=True)
_recorder_file = RECORDER_DIR / "index.html"
if not _recorder_file.exists() or _recorder_file.read_text(encoding="utf-8") != RECORDER_HTML:
    _recorder_file.write_text(RECORDER_HTML, encoding="utf-8")

voice_recorder = components.declare_component("voice_recorder", path=str(RECORDER_DIR))

st.set_page_config(page_title=APP_NAME, page_icon="🎙️", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, button, input, textarea, select {font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}
.block-container {padding: 1.1rem 1rem 3rem 1rem; max-width: 900px;}

.topbar {display: flex; align-items: center; gap: 14px; background: #0f172a; border-radius: 18px; padding: 16px 20px; margin-bottom: 18px;}
.logo {width: 46px; height: 46px; border-radius: 13px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; color: #ffffff; background: linear-gradient(135deg, #2563eb, #4f46e5);}
.brand {color: #ffffff; font-weight: 700; font-size: 1.3rem; line-height: 1.2; letter-spacing: -0.2px;}
.tagline {color: #94a3b8; font-size: 0.85rem; margin-top: 2px;}

.sec {display: flex; align-items: center; gap: 12px; margin-bottom: 12px;}
.sec .ico {width: 38px; height: 38px; border-radius: 11px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; color: #2563eb; background: rgba(37, 99, 235, 0.12);}
.sec .t {font-weight: 650; font-size: 1.05rem; line-height: 1.2;}
.sec .s {font-size: 0.82rem; opacity: 0.65; margin-top: 2px;}

div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}

div[data-baseweb="tab-list"] {gap: 6px; background: rgba(127, 127, 127, 0.13); padding: 5px; border-radius: 13px;}
button[data-baseweb="tab"] {flex: 1; border-radius: 9px; height: auto; padding: 9px 16px;}
button[data-baseweb="tab"] p {font-weight: 600; font-size: 0.97rem;}
button[data-baseweb="tab"][aria-selected="true"] {background: #2563eb;}
button[data-baseweb="tab"][aria-selected="true"] p {color: #ffffff;}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display: none;}

div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] > div {border-radius: 10px;}

.stButton > button, .stDownloadButton > button {width: 100%; border-radius: 12px; font-weight: 600; padding: 0.7rem 1rem; border: 1px solid rgba(127, 127, 127, 0.32); transition: transform 0.12s ease, box-shadow 0.12s ease;}
.stButton > button:hover, .stDownloadButton > button:hover {transform: translateY(-1px); box-shadow: 0 6px 16px rgba(37, 99, 235, 0.22);}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {color: #ffffff; border: none; font-size: 1.02rem; padding: 0.85rem 1rem; background: linear-gradient(135deg, #2563eb, #4f46e5);}
.stButton > button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p {color: #ffffff;}
.stButton > button[kind="primary"]:disabled, button[data-testid="stBaseButton-primary"]:disabled {opacity: 0.4;}

div[data-testid="stMetric"] {background: rgba(127, 127, 127, 0.09); border: 1px solid rgba(127, 127, 127, 0.2); padding: 14px 18px; border-radius: 14px;}
div[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.7;}

.empty {text-align: center; padding: 42px 16px; border: 1px dashed rgba(127, 127, 127, 0.4); border-radius: 16px;}
.empty .ico {display: inline-flex; width: 54px; height: 54px; border-radius: 16px; align-items: center; justify-content: center; color: #2563eb; background: rgba(37, 99, 235, 0.12); margin-bottom: 10px;}
.empty .t {font-weight: 650; font-size: 1.05rem;}
.empty .s {font-size: 0.88rem; opacity: 0.65; margin-top: 4px;}
.note {text-align: center; font-size: 0.8rem; opacity: 0.55; margin-top: 26px;}
</style>
""",
    unsafe_allow_html=True,
)

ICONS = {
    "mic": '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "folder": '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
}


def svg(name, size=20):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{ICONS[name]}</svg>'
    )


def section(icon, title, subtitle):
    st.markdown(
        f'<div class="sec"><div class="ico">{svg(icon)}</div>'
        f'<div><div class="t">{title}</div><div class="s">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    f'<div class="topbar"><div class="logo">{svg("mic", 24)}</div>'
    f'<div><div class="brand">{APP_NAME}</div>'
    f'<div class="tagline">Record, transcribe and summarize your meetings</div></div></div>',
    unsafe_allow_html=True,
)

tab_new, tab_history = st.tabs(["New recording", "Library"])

with tab_new:
    with st.container(border=True):
        section("mic", "Audio", "Record live or upload a file")
        rec_col, up_col = st.columns(2)
        with rec_col:
            st.caption("Record")
            recorded = voice_recorder(key="voice_recorder", default=None)
        with up_col:
            st.caption("Upload")
            uploaded = st.file_uploader(
                "Upload",
                type=["mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac"],
                label_visibility="collapsed",
            )

        audio_bytes, filename = None, None
        if uploaded is not None:
            audio_bytes, filename = uploaded.getvalue(), uploaded.name
        elif isinstance(recorded, dict) and recorded.get("audio"):
            audio_bytes = base64.b64decode(recorded["audio"])
            filename = f"recording.{recorded.get('ext', 'webm')}"

        if audio_bytes:
            size_mb = len(audio_bytes) / (1024 * 1024)
            if size_mb > MAX_MB:
                st.error(
                    f"This file is {size_mb:.1f} MB, which is larger than the {MAX_MB} MB limit. "
                    "Split it into shorter parts or compress it first."
                )
                audio_bytes = None
            else:
                st.success(f"Audio ready: {filename} ({size_mb:.1f} MB)")

    with st.container(border=True):
        section("sliders", "Details", "Helps the transcript and notes come out right")
        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input("Title")
            language = st.selectbox("Language spoken", ["Auto-detect", "English", "Hindi", "Marathi"])
        with c2:
            vocab = st.text_area(
                "Names and special words",
                height=122,
                help="Add names, brands and technical terms that may be heard wrongly. This improves spelling.",
            )
        context = st.text_input(
            "Context for the notes",
            help="Optional. One short line about what this recording is.",
        )

    go = st.button(
        "Transcribe and write notes",
        type="primary",
        icon=":material/auto_awesome:",
        disabled=audio_bytes is None,
    )

    if go:
        with st.status("Processing your recording", expanded=True) as status:
            try:
                st.write("Transcribing audio...")
                transcript = transcribe(audio_bytes, filename, language, vocab)
                if not transcript.strip():
                    status.update(label="No speech found", state="error", expanded=True)
                    st.warning("No speech was found in this audio.")
                else:
                    st.write("Writing notes...")
                    notes = make_notes(transcript, context, progress=st.write)
                    st.session_state["result"] = {
                        "title": title.strip() or "Untitled recording",
                        "transcript": transcript,
                        "notes": notes,
                    }
                    status.update(label="Finished", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Something went wrong", state="error", expanded=True)
                st.error(str(e))

    result = st.session_state.get("result")
    if result:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        section("check", "Results", "Check the notes against the transcript before relying on them")

        stamps = re.findall(r"\[(\d+):(\d\d)\]", result["transcript"])
        length_text = f"{int(stamps[-1][0])}:{int(stamps[-1][1]):02d}" if stamps else "n/a"
        words = len(re.sub(r"\[\d+:\d\d\]", "", result["transcript"]).split())

        m1, m2, m3 = st.columns(3)
        m1.metric("Length", length_text)
        m2.metric("Words", f"{words:,}")
        m3.metric("Segments", len(result["transcript"].splitlines()))

        with st.container(border=True):
            section("list", "Notes", result["title"])
            st.markdown(result["notes"])

        with st.container(border=True):
            section("file", "Transcript", "Full text with time stamps")
            st.text_area(
                "Full transcript",
                result["transcript"],
                height=360,
                label_visibility="collapsed",
            )

        b1, b2, b3 = st.columns(3)
        if b1.button("Save", icon=":material/save:"):
            save_meeting(result["title"], result["transcript"], result["notes"])
            st.success("Saved to your Library.")
        b2.download_button(
            "Notes",
            result["notes"],
            file_name=f"{result['title']}_notes.md",
            icon=":material/download:",
        )
        b3.download_button(
            "Transcript",
            result["transcript"],
            file_name=f"{result['title']}_transcript.txt",
            icon=":material/download:",
        )

with tab_history:
    rows = load_meetings()
    if not rows:
        st.markdown(
            f'<div class="empty"><div class="ico">{svg("folder", 26)}</div>'
            '<div class="t">No saved recordings yet</div>'
            '<div class="s">Recordings you save will appear here.</div></div>',
            unsafe_allow_html=True,
        )
    for mid, created, mtitle, mtranscript, mnotes in rows:
        with st.expander(f"{mtitle}  |  {created}"):
            st.markdown(mnotes)
            st.text_area("Transcript", mtranscript, height=240, key=f"t{mid}")
            c1, c2 = st.columns(2)
            c1.download_button(
                "Download notes",
                mnotes,
                file_name=f"{mtitle}_notes.md",
                key=f"d{mid}",
                icon=":material/download:",
            )
            if c2.button("Delete", key=f"x{mid}", icon=":material/delete:"):
                delete_meeting(mid)
                st.rerun()

st.markdown(
    '<div class="note">Always check important notes against the transcript.</div>',
    unsafe_allow_html=True,
)
