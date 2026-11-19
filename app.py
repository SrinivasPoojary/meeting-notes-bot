import os
import sqlite3
import datetime

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


def make_notes_groq(user_text):
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it in secrets.")
    last_error = "No notes model is available on this Groq account."
    for model in notes_model_candidates(key):
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 4000,
            "messages": [
                {"role": "system", "content": NOTES_SYSTEM},
                {"role": "user", "content": user_text},
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
            continue
        last_error = f"{model}: {resp.status_code} {resp.text[:200]}"
        # Try the next model if this one is missing or not allowed
        if resp.status_code in (400, 403, 404):
            continue
        break
    raise RuntimeError(f"Notes failed. Last error: {last_error}")


def make_notes(transcript, context):
    user_text = ""
    if context.strip():
        user_text += f"Meeting context from the user:\n{context.strip()}\n\n"
    user_text += f"Transcript:\n{transcript}"

    key = get_secret("ANTHROPIC_API_KEY")
    if not key:
        # Free option: Groq writes the notes when no Anthropic key is set
        return make_notes_groq(user_text)

    import anthropic

    model = get_secret("CLAUDE_MODEL") or "claude-sonnet-5-5"
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0,
        system=NOTES_SYSTEM,
        messages=[{"role": "user", "content": user_text}],
    )
    return msg.content[0].text


# ---------- screen ----------
import re

st.set_page_config(page_title="Meeting Notes Bot", page_icon="🎙️", layout="wide")

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1100px;}

.hero {
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 45%, #ec4899 100%);
  border-radius: 24px; padding: 30px 32px; margin-bottom: 22px;
  box-shadow: 0 14px 34px rgba(124, 58, 237, 0.35);
}
.hero-title {font-size: 2.3rem; font-weight: 800; color: #ffffff; line-height: 1.15;}
.hero-sub {font-size: 1.05rem; color: rgba(255,255,255,0.92); margin: 8px 0 16px 0;}
.chip {
  display: inline-block; padding: 6px 14px; margin: 0 8px 8px 0; border-radius: 999px;
  background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.35);
  color: #ffffff; font-size: 0.88rem; font-weight: 600;
}

.step {display: flex; align-items: center; gap: 12px; margin: 4px 0 10px 0;}
.step .num {
  width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
  background: linear-gradient(135deg, #6366f1, #ec4899); color: #ffffff;
  display: flex; align-items: center; justify-content: center; font-weight: 800;
  box-shadow: 0 4px 12px rgba(236, 72, 153, 0.35);
}
.step .label {font-size: 1.25rem; font-weight: 700;}

div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 18px;}

.stButton > button, .stDownloadButton > button {
  width: 100%; border-radius: 12px; font-weight: 600; padding: 0.6rem 1rem;
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px); box-shadow: 0 8px 18px rgba(99, 102, 241, 0.28);
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #6366f1, #8b5cf6 50%, #ec4899);
  color: #ffffff; border: none; font-size: 1.05rem; padding: 0.75rem 1rem;
}
.stButton > button[kind="primary"]:disabled, button[data-testid="stBaseButton-primary"]:disabled {
  opacity: 0.45;
}

div[data-testid="stMetric"] {
  background: rgba(127, 127, 127, 0.10); border: 1px solid rgba(127, 127, 127, 0.22);
  padding: 14px 18px; border-radius: 16px;
}
button[data-baseweb="tab"] {font-size: 1.05rem; font-weight: 700;}
.footer {text-align: center; opacity: 0.6; font-size: 0.9rem; margin-top: 28px;}
</style>
""",
    unsafe_allow_html=True,
)


def step(number, text):
    st.markdown(
        f'<div class="step"><div class="num">{number}</div><div class="label">{text}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    """
<div class="hero">
<div class="hero-title">🎙️ Meeting Notes Bot</div>
<div class="hero-sub">Record or upload any meeting or call. Get an accurate transcript and clean notes in seconds.</div>
<span class="chip">🎧 Voice to text</span>
<span class="chip">✨ Smart notes</span>
<span class="chip">✅ Action items</span>
<span class="chip">💾 Save &amp; download</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_new, tab_history = st.tabs(["🎙️ New meeting", "📚 Saved meetings"])

with tab_new:
    with st.container(border=True):
        step(1, "📝 Meeting details")
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("🏷️ Meeting title", placeholder="Weekly sales review")
            language = st.selectbox(
                "🌐 Language spoken", ["Auto-detect", "English", "Hindi", "Marathi"]
            )
        with col2:
            vocab = st.text_area(
                "🔤 Names and special words (improves accuracy)",
                placeholder="Responsive Industries, Power BI, DAX, Srinivas",
                height=122,
            )
        context = st.text_input(
            "💡 Short context for the notes (optional)", placeholder="Call with the export team"
        )

    with st.container(border=True):
        step(2, "🎧 Add your audio")
        rec_col, up_col = st.columns(2)
        with rec_col:
            st.markdown("**🎤 Record live**")
            recorded = st.audio_input("Press the mic to start", label_visibility="collapsed")
        with up_col:
            st.markdown("**📁 Upload a recording**")
            uploaded = st.file_uploader(
                "Upload a recording",
                type=["mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac"],
                label_visibility="collapsed",
            )
            st.caption("mp3, wav, m4a, mp4, webm  ·  up to 25 MB")

        audio_bytes, filename = None, None
        if uploaded is not None:
            audio_bytes, filename = uploaded.getvalue(), uploaded.name
        elif recorded is not None:
            audio_bytes, filename = recorded.getvalue(), "recording.wav"

        if audio_bytes:
            size_mb = len(audio_bytes) / (1024 * 1024)
            if size_mb > MAX_MB:
                st.error(
                    f"⚠️ This file is {size_mb:.1f} MB, larger than {MAX_MB} MB. "
                    "Split it into shorter parts or compress it first."
                )
                audio_bytes = None
            else:
                st.success(f"🎵 Audio ready: {filename}  ({size_mb:.1f} MB)")

    with st.container(border=True):
        step(3, "✨ Create your notes")
        go = st.button(
            "Transcribe and write notes",
            type="primary",
            icon=":material/auto_awesome:",
            disabled=audio_bytes is None,
        )
        if audio_bytes is None:
            st.caption("Add audio in step 2 to unlock this button.")

        if go:
            try:
                with st.spinner("🎧 Listening to the audio..."):
                    transcript = transcribe(audio_bytes, filename, language, vocab)
                if not transcript.strip():
                    st.warning("No speech was found in this audio.")
                else:
                    with st.spinner("✍️ Writing the notes..."):
                        notes = make_notes(transcript, context)
                    st.session_state["result"] = {
                        "title": title.strip() or "Untitled meeting",
                        "transcript": transcript,
                        "notes": notes,
                    }
            except Exception as e:
                st.error(str(e))

    result = st.session_state.get("result")
    if result:
        st.success("🎉 Done! Check the notes against the transcript before relying on them.")

        stamps = re.findall(r"\[(\d+):(\d\d)\]", result["transcript"])
        if stamps:
            last_min, last_sec = int(stamps[-1][0]), int(stamps[-1][1])
            length_text = f"{last_min}:{last_sec:02d}"
        else:
            length_text = "n/a"
        words = len(re.sub(r"\[\d+:\d\d\]", "", result["transcript"]).split())

        m1, m2, m3 = st.columns(3)
        m1.metric("⏱️ Approx. length", length_text)
        m2.metric("💬 Words spoken", f"{words:,}")
        m3.metric("🧾 Transcript lines", len(result["transcript"].splitlines()))

        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("### 📝 Notes")
                st.markdown(result["notes"])
        with right:
            with st.container(border=True):
                st.markdown("### 🗒️ Transcript")
                st.text_area(
                    "Full transcript",
                    result["transcript"],
                    height=420,
                    label_visibility="collapsed",
                )

        b1, b2, b3 = st.columns(3)
        if b1.button("Save this meeting", icon=":material/save:"):
            save_meeting(result["title"], result["transcript"], result["notes"])
            st.success("💾 Saved. Find it in the Saved meetings tab.")
        b2.download_button(
            "Download notes",
            result["notes"],
            file_name=f"{result['title']}_notes.md",
            icon=":material/description:",
        )
        b3.download_button(
            "Download transcript",
            result["transcript"],
            file_name=f"{result['title']}_transcript.txt",
            icon=":material/download:",
        )

with tab_history:
    rows = load_meetings()
    if not rows:
        st.info("📭 No saved meetings yet. Create one in the New meeting tab and press Save.")
    for mid, created, mtitle, mtranscript, mnotes in rows:
        with st.expander(f"📝 {mtitle}   ·   🗓️ {created}"):
            st.markdown(mnotes)
            st.text_area("Transcript", mtranscript, height=250, key=f"t{mid}")
            c1, c2 = st.columns(2)
            c1.download_button(
                "Download notes",
                mnotes,
                file_name=f"{mtitle}_notes.md",
                key=f"d{mid}",
                icon=":material/description:",
            )
            if c2.button("Delete", key=f"x{mid}", icon=":material/delete:"):
                delete_meeting(mid)
                st.rerun()

st.markdown(
    '<div class="footer">🔒 Your audio is sent only to the speech and notes services. '
    "Always check important notes against the transcript.</div>",
    unsafe_allow_html=True,
)
