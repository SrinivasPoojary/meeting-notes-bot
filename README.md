# Meeting Notes Bot

Record or upload a meeting or call. The app writes a timestamped transcript and structured notes (summary, key points, decisions, action items, open questions).

## How it works
1. Audio goes to Groq Whisper (`whisper-large-v3`) for speech-to-text.
2. The transcript goes to an AI model (Groq by default, or Claude if you add an Anthropic key), which writes notes using only what was said.
3. Transcript and notes are saved in a local SQLite file and can be downloaded.

## Keys you need
- `GROQ_API_KEY` from console.groq.com (free tier available). This alone is enough: Groq does the speech-to-text and also writes the notes. Cost: free.
- `ANTHROPIC_API_KEY` from console.anthropic.com (pay as you go). Optional. If you add it, Claude writes the notes instead, which is usually more accurate.

## Run on your computer
1. Install Python 3.10 or newer.
2. Run: `pip install -r requirements.txt`
3. Create the file `.streamlit/secrets.toml` with:

```
GROQ_API_KEY = "your-groq-key"
```

(Optional, paid) add this line to use Claude for the notes:

```
ANTHROPIC_API_KEY = "your-anthropic-key"
```

4. Run: `streamlit run app.py`

## Put it online (Streamlit Community Cloud)
1. Upload `app.py`, `requirements.txt` and `README.md` to a GitHub repository. Do NOT upload `secrets.toml`.
2. Go to share.streamlit.io, sign in with GitHub and choose New app.
3. Pick your repository and set the main file to `app.py`.
4. Open Advanced settings, then Secrets, and paste your Groq key in the same format as above.
5. Deploy, then open the link on your phone.

## Good to know
- Saved meetings on Streamlit Cloud can be wiped when the app restarts. Always use the download buttons for notes you want to keep.
- Uploads are limited to 25 MB. Split longer recordings.
- Phone calls: the microphone only hears what plays through the speaker. For best accuracy, record the call with your phone's call recorder or a meeting tool, then upload the file.
- Add names and special words in the vocabulary box. This improves spelling.
