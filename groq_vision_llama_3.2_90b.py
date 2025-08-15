import os
import base64
from groq import Groq
from dotenv import load_dotenv


def load_api_key():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("API key no encontrada en el archivo .env.")
    return api_key


def initialize_client(api_key):
    return Groq(api_key=api_key)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_user_text_input():
    return input("\n¿Qué le quieres preguntar a Llama 3.2-90B-Visión?: ")


def get_user_image_input():
    intentos = 0
    max_intentos = 3
    
    while intentos < max_intentos:
        choice = input("Tipea 'url' si quieres usar un link de internet o 'ruta' si quieres subir una imagen desde tu dispositivo: ").strip("'").lower()
        if choice in ['url', 'ruta']:
            if choice == 'url':
                return input("Ingresa la url de tu imagen: ")
            if choice == 'ruta':
                image_path = input("Ingresa la ruta completa de tu imagen: ").strip('"')
                base64_image = encode_image(image_path)
                return f"data:image/jpeg;base64,{base64_image}"
        else:
            intentos += 1
            intentos_restantes = max_intentos - intentos
            if intentos_restantes > 0:
                print(f"Opción no válida")
            else:
                raise ValueError("El usuario no ha proporcionado una opción correcta hasta 3 veces. Programa terminado.\n")


def get_image_analysis(client, text_input, image_input):
    stream = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "assistant",
                "content": ""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_input
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_input
                        }
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        stream=True,
        stop=None,
    )
    return stream

def main():
    try:
        api_key = load_api_key()
        client = initialize_client(api_key)
        text_input = get_user_text_input()
        image_input = get_user_image_input()
        stream = get_image_analysis(client, text_input, image_input)
        print("________________________________________________\n")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print("\n________________________________________________\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
