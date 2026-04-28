import os
import google.generativeai as genai
from google.cloud import texttospeech

# 1. Initialize Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("[!] Warning: GEMINI_API_KEY environment variable not set!")
else:
    genai.configure(api_key=api_key)


def generate_bilingual_alert(fno_output):
    print("Transmitting telemetry to Gemini 2.0 Flash...")

    prompt = f"""
    You are an emergency flood broadcast system for the Damodar River Basin. 
    Based on this telemetry data: {fno_output}
    Generate a short, urgent bilingual broadcast. 
    1. English first (1 sentence).
    2. Bengali translation second (1-2 sentences). 
    The Bengali MUST be plain-language, conversational, and actionable for rural farmers in Ghatal. Do not use overly formal academic Bengali.
    DO NOT include markdown, asterisks, or labels like 'English:'. Just the raw spoken text.
    """

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        text_output = response.text.strip()
        print(f"\n[LIVE BROADCAST SCRIPT]\n{text_output}\n")
        return text_output

    except Exception as e:
        print("\n[!] GEMINI API RATE LIMIT HIT. TRIGGERING CACHED FALLBACK SCRIPT [!]")
        fallback_text = (
            "Warning: Critical flood levels detected in Ghatal. Please evacuate to higher ground immediately. "
            "সতর্কবার্তা: ঘাটালে বন্যার জল বিপদসীমার ওপরে। দয়া করে এখনই উঁচু জায়গায় চলে যান। আপনার পরিবারকে সুরক্ষিত রাখুন।"
        )
        print(f"\n[CACHED BROADCAST SCRIPT]\n{fallback_text}\n")
        return fallback_text


def create_audio_alert(text_output):
    print("Synthesizing Neural Bengali Audio via Google Cloud TTS...")
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text_output)

        voice = texttospeech.VoiceSelectionParams(
            language_code="bn-IN",
            name="bn-IN-Wavenet-A"
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        filename = "ghatal_alert.mp3"
        with open(filename, "wb") as out:
            out.write(response.audio_content)
            print(f"[+] Success: Audio saved locally as '{filename}'")
    except Exception as e:
        print(f"[!] Critical Error generating audio. Is Google Cloud authenticated? Error: {e}")


# --- THIS IS THE EXECUTION BLOCK THAT WAS MISSING ---
if __name__ == "__main__":
    dummy_fno_json = {
        "location": "Ghatal",
        "danger_level": "CRITICAL",
        "water_depth": 4.27,
        "time_to_peak": 14,
        "confidence_pct": 87
    }

    script = generate_bilingual_alert(dummy_fno_json)
    create_audio_alert(script)