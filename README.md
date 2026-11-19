# AI Voice Recording

**Live app:** [Open AI Voice Recording](https://YOUR-APP-NAME.streamlit.app)

AI Voice Recording turns meetings and calls into a written transcript and clear notes. Record your voice in the browser or upload an audio file, and the app returns the full text with time stamps plus a summary, key points, decisions and action items.

## What it does
- **Records live** in the browser with one Record / Pause / Resume button, a Stop button and a running timer.
- **Accepts uploads** in mp3, wav, m4a, mp4, webm, ogg and flac.
- **Transcribes speech** in English, Hindi and Marathi, or detects the language automatically.
- **Writes structured notes:** summary, key points, decisions made, action items and open questions.
- **Handles long meetings** by writing notes in parts and combining them into one final set.
- **Saves a library** of recordings so you can reopen notes and transcripts later.
- **Downloads** notes (.md) and transcripts (.txt).

## How to use
1. Open the app link above. Allow the microphone when the browser asks.
2. In **Audio**, press **Record** to start. Use **Pause** and **Resume** as needed, then press **Stop**. You can also upload a recording instead.
3. In **Details**, add a title and choose the language. Optionally add names and special words, which improves spelling, and a short context line for the notes.
4. Press **Transcribe and write notes**.
5. Read the notes and check them against the transcript, then press **Save** or download them.

## Tips for accurate results
- Record in a quiet place and keep the microphone close to the speaker.
- Let one person speak at a time.
- Add names, brands and technical terms in **Names and special words**.
- Choose the spoken language instead of auto-detect when you know it.
- For phone calls, use your phone's call recorder and upload the file. The browser microphone only hears what plays through the speaker.
- Always check important notes against the transcript. No automatic tool is perfect.

## Limits on the free setup
- Each audio file can be up to **25 MB**. The recorder saves speech at a low bitrate, so one recording can run about 1.5 hours (estimate).
- Groq's free tier has daily and per-minute limits. Long meetings are processed in parts, so notes for a 90-minute meeting can take a few minutes. If you hit a limit, wait a while and try again.
- Saved recordings are kept in a small database file. On Streamlit Community Cloud this file can be erased when the app restarts, so download anything important.

## How it works
1. Audio goes to Groq Whisper (`whisper-large-v3`) for speech-to-text, with time stamps.
2. The transcript goes to a language model that writes the notes using only what was said. Unclear parts are marked `[unclear]`.
   - By default the free Groq model is used and the app picks one that is available on your account.
   - If you add an Anthropic key, Claude writes the notes instead.
3. Transcripts and notes are stored in a local SQLite file (`meetings.db`).

## Keys
| Key | Required | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | Yes | console.groq.com (free tier) |
| `ANTHROPIC_API_KEY` | No | console.anthropic.com (paid, optional) |

Optional settings: `GROQ_NOTES_MODEL` to force a specific Groq notes model, and `CLAUDE_MODEL` to choose a Claude model.

## Run on your computer
1. Install Python 3.10 or newer.
2. Install the packages: `pip install -r requirements.txt`
3. Create the file `.streamlit/secrets.toml` and add your key:

```
GROQ_API_KEY = "your-groq-key"
```

4. Start the app: `streamlit run app.py`

## Put it online (Streamlit Community Cloud)
1. Upload `app.py`, `requirements.txt` and `README.md` to a public GitHub repository. Never upload `secrets.toml`.
2. Go to share.streamlit.io, sign in with GitHub and choose **Create app**.
3. Select your repository, branch `main` and main file `app.py`.
4. Open **Advanced settings**, then **Secrets**, and paste your key in the format shown above.
5. Click **Deploy**, then open the link on your phone or computer.

## Project files
- `app.py` - the whole app: design, recorder, transcription and notes.
- `requirements.txt` - Python packages.
- `README.md` - this guide.

The app creates a `voice_recorder_component` folder and a `meetings.db` file by itself while it runs. Do not upload them to GitHub.

## Privacy
Audio and transcripts are sent to Groq (and to Anthropic if you add that key) to produce the text and notes. Do not use the app for recordings you are not allowed to share with those services.
