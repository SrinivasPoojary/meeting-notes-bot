import os
import sqlite3
import datetime

import requests
import streamlit as st
import anthropic

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
def make_notes(transcript, context):
    key = get_secret("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it in secrets.")
    model = get_secret("CLAUDE_MODEL") or "claude-sonnet-5-5"
    client = anthropic.Anthropic(api_key=key)
    user_text = ""
    if context.strip():
        user_text += f"Meeting context from the user:\n{context.strip()}\n\n"
    user_text += f"Transcript:\n{transcript}"
    msg = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0,
        system=NOTES_SYSTEM,
        messages=[{"role": "user", "content": user_text}],
    )
    return msg.content[0].text


# ---------- screen ----------
st.set_page_config(page_title="Meeting Notes Bot", page_icon="🎙️", layout="wide")
st.title("🎙️ Meeting Notes Bot")
st.caption("Record or upload a meeting. It writes the transcript and the notes.")

tab_new, tab_history = st.tabs(["New meeting", "Saved meetings"])

with tab_new:
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Meeting title", placeholder="Example: Weekly sales review")
        language = st.selectbox("Language spoken", ["Auto-detect", "English", "Hindi", "Marathi"])
    with col2:
        vocab = st.text_area(
            "Names and special words (improves accuracy)",
            placeholder="Example: Responsive Industries, Power BI, DAX, Srinivas",
            height=100,
        )
    context = st.text_input("Short context for the notes (optional)", placeholder="Example: Call with the export team")

    st.subheader("1. Add the audio")
    recorded = st.audio_input("Record from your microphone")
    uploaded = st.file_uploader(
        "Or upload a recording (mp3, wav, m4a, mp4, webm)",
        type=["mp3", "wav", "m4a", "mp4", "webm", "ogg", "flac"],
    )

    audio_bytes, filename = None, None
    if uploaded is not None:
        audio_bytes, filename = uploaded.getvalue(), uploaded.name
    elif recorded is not None:
        audio_bytes, filename = recorded.getvalue(), "recording.wav"

    if audio_bytes:
        size_mb = len(audio_bytes) / (1024 * 1024)
        st.write(f"Audio size: {size_mb:.1f} MB")
        if size_mb > MAX_MB:
            st.error(f"This file is larger than {MAX_MB} MB. Split it into shorter parts or compress it first.")
            audio_bytes = None

    st.subheader("2. Create transcript and notes")
    if st.button("Transcribe and write notes", type="primary", disabled=audio_bytes is None):
        try:
            with st.spinner("Listening to the audio..."):
                transcript = transcribe(audio_bytes, filename, language, vocab)
            if not transcript.strip():
                st.warning("No speech was found in this audio.")
            else:
                with st.spinner("Writing the notes..."):
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
        st.success("Done. Check the notes against the transcript before relying on them.")
        left, right = st.columns(2)
        with left:
            st.markdown("### Notes")
            st.markdown(result["notes"])
        with right:
            st.markdown("### Transcript")
            st.text_area("Full transcript", result["transcript"], height=400, label_visibility="collapsed")

        b1, b2, b3 = st.columns(3)
        if b1.button("Save this meeting"):
            save_meeting(result["title"], result["transcript"], result["notes"])
            st.success("Saved.")
        b2.download_button(
            "Download notes",
            result["notes"],
            file_name=f"{result['title']}_notes.md",
        )
        b3.download_button(
            "Download transcript",
            result["transcript"],
            file_name=f"{result['title']}_transcript.txt",
        )

with tab_history:
    rows = load_meetings()
    if not rows:
        st.info("No saved meetings yet.")
    for mid, created, mtitle, mtranscript, mnotes in rows:
        with st.expander(f"{created}  |  {mtitle}"):
            st.markdown(mnotes)
            st.text_area("Transcript", mtranscript, height=250, key=f"t{mid}")
            c1, c2 = st.columns(2)
            c1.download_button("Download notes", mnotes, file_name=f"{mtitle}_notes.md", key=f"d{mid}")
            if c2.button("Delete", key=f"x{mid}"):
                delete_meeting(mid)
                st.rerun()
