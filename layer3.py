import os
import json
import google.generativeai as genai
from google.cloud import texttospeech

# 1. Setup API Keys (You will need to set these in your terminal or a .env file)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAJ6cOKfxnsrwX9atWN7NgXAGOsPF5b_7U")
genai.configure(api_key=GEMINI_API_KEY)


def generate_bengali_alert(json_payload):
    print("Initializing Gemini 2.0 Flash Multimodal Pipeline...")

    # We specifically use Gemini 2.0 Flash as per the architecture doc
    model = genai.GenerativeModel('gemini-2.5-flash')  # Using the latest flash variant

    prompt = f"""
    You are an emergency flood warning system for the Damodar River Basin in West Bengal.
    Take this technical JSON flood prediction and convert it into a 2-sentence, urgent, 
    highly actionable Bengali warning message for the local farmers in Ghatal.

    JSON Data:
    {json.dumps(json_payload)}

    Rules:
    1. Output ONLY the Bengali text. No English, no explanations.
    2. Mention the exact danger level and time to peak.
    3. End with a clear instruction to move to higher ground.
    """

    response = model.generate_content(prompt)
    bengali_text = response.text.strip()
    print(f"Gemini Output: {bengali_text}")
    return bengali_text


def synthesize_audio(text, output_filename="alert.mp3"):
    print("Connecting to Google Cloud TTS (Neural2 Bengali Voice)...")

    # Note: This requires Google Cloud Credentials to be set up in your environment
    try:
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Select the exact Neural2 Bengali voice specified in the pitch
        voice = texttospeech.VoiceSelectionParams(
            language_code="bn-IN",
            name="bn-IN-Wavenet-A"  # High-quality Bengali voice
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_filename, "wb") as out:
            out.write(response.audio_content)
            print(f"Audio successfully saved to {output_filename}")

    except Exception as e:
        print("\n[!] Google Cloud TTS Error. Make sure your GOOGLE_APPLICATION_CREDENTIALS are set.")
        print(f"Error details: {e}")


if __name__ == "__main__":
    # Mock payload from your successful FastAPI test
    mock_payload = {
        "prediction": {
            "location": "Ghatal, West Bengal",
            "danger_level": "CRITICAL",
            "water_depth_meters": 6.87,
            "confidence_pct": 88.5,
            "time_to_peak_hours": 14
        }
    }

    # 1. Generate Text
    bengali_alert = generate_bengali_alert(mock_payload)

    # 2. Generate Audio
    synthesize_audio(bengali_alert, "ghatal_warning.mp3")