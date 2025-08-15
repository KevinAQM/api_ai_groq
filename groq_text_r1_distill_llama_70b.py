import os
from gtts import gTTS
from dotenv import load_dotenv
from groq import Groq


def load_api_key():
    """
    Loads the Groq API key from the .env file.

    Raises:
        ValueError: If the GROQ_API_KEY is not found in the .env file.

    Returns:
        str: The Groq API key.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("API key no encontrada en el archivo .env.")
    return api_key


def initialize_client(api_key):
    """
    Initializes and returns a Groq client instance using the provided API key.

    Args:
        api_key (str): The Groq API key.

    Returns:
        Groq: An initialized Groq client instance.
    """
    return Groq(api_key=api_key)


def get_user_input():
    """
    Prompts the user for input to ask the language model.

    Returns:
        str: The user's input string.
    """
    return input("\n¿Qué le quieres preguntar a Llama 3.3-70B-Text?: ")


def generate_response(client, user_input):
    """
    Generates a text response from the Groq API using the specified model.

    Args:
        client (Groq): The initialized Groq client.
        user_input (str): The user's question or prompt.

    Returns:
        str: The generated text response from the model.
    """
    stream = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        temperature=0.5,
        max_tokens=4096,
        top_p=0.95,
        stream=True,
        stop=None,
        # reasoning_format="hidden"
    )
    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    return response


def text_to_speech(text, language):
    """
    Converts the given text to speech using gTTS and saves/plays the audio.

    Args:
        text (str): The text to convert to speech.
        language (str): The language code for the speech ('es' for Spanish, 'en' for English).
    """
    try:
        gTTS.GOOGLE_TTS_MAX_CHARS = 200 # Limit characters for gTTS free tier if needed
        tts = gTTS(text=text, lang=language)
        audio_file = "response.mp3"
        tts.save(audio_file)
        print(f"Audio guardado como {audio_file}. Reproduciendo...\n")
        os.system(f"start {audio_file}")  # For Windows. Use 'afplay' or 'mpg123' on macOS/Linux
    except Exception as e:
        print(f"Error al convertir texto a voz: {e}")


def main():
    """
    Main function to orchestrate the process:
    1. Load API key.
    2. Initialize Groq client.
    3. Get user input for the prompt and desired language.
    4. Generate text response using Groq API.
    5. Print the text response.
    6. Convert the text response to speech and play it.
    """
    try:
        api_key = load_api_key()
        client = initialize_client(api_key)
        user_input = get_user_input()
        language_choice = input("¿Quieres la respuesta en voz española (es) o inglesa (en)?: ").strip().lower()
        language = "es" if language_choice == "es" else "en"
        print("Generando respuesta en texto...\n")
        print("-------------------------------------")
        response = generate_response(client, user_input)
        print(f"{response}")
        print("-------------------------------------\n")
        print("Generando respuesta en audio...")
        text_to_speech(response, language)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
