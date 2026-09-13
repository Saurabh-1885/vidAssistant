import yt_dlp
from pydub import AudioSegment
import os

# Creating a folder to save audio downloaded from YouTube
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    # Configuration for downloading audio from YouTube
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],

        "quiet": True,
    }

    # Download the audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # Get the original filename
        filename = ydl.prepare_filename(info).replace(".webm",".wav").replace(".m4a",".wav")

    return filename




#Convert to wav function

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path) [0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)  #AudioSegment automatically detects type of file (eg. mp4,m4a,etc)
    audio = audio.set_channels(1).set_frame_rate(16000) #Sets the audio to monoaudio and quality to 16 kHz
    audio. export (output_path, format="wav")
    return output_path 



#Converting to chunks

def chunk_audio(wav_path :str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)

    return chunks

#Creating input function

def process_input(source: str)-> list:
    if source.startswith("https://") or source.startswith("https://"):
        print("YouTube URL detected, Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Local file detected, Downloading audio...")
        wav_path = convert_to_wav(source)

    print("Checking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks        



