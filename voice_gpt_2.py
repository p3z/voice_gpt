# origin: https://github.com/Ai-Austin/Bing-GPT-Voice-Assistant/blob/main/main.py
import openai
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize the OpenAI API
# Initialize the OpenAI API
openai.api_key = os.getenv('OPENAI_KEY')

print("Loading voiceGPT")
# print("API key: ")
# print(openai.api_key_path)
# print(os.getenv('OPENAI_KEY'))

import sys
import ffmpeg
import pyttsx3
import pyaudio
import asyncio
import re
import whisper # not the pip package, but is related to openai (https://github.com/openai/whisper/blob/main/README.md)
import boto3
import time
import pydub
from pydub import playback
import speech_recognition as sr
from EdgeGPT import Chatbot, ConversationStyle
import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
from datetime import datetime


os.close(sys.stderr.fileno()) # hide errors



print("Loading voiceGPT")
##print("API key: ")
#print(openai.api_key_path)

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003", #gpt-3.5-turbo to try
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"]

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()
BING_WAKE_WORD = "bing"
GPT_WAKE_WORD = "chat"

# Initialize the text to speech engine 
engine = pyttsx3.init()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def get_wake_word(phrase):
    if BING_WAKE_WORD in phrase.lower():
        return BING_WAKE_WORD
    elif GPT_WAKE_WORD in phrase.lower():
        return GPT_WAKE_WORD
    else:
        return None
    
def synthesize_speech(text, output_filename):
    polly = boto3.client('polly', region_name='us-west-2')
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Salli',
        Engine='neural'
    )

    with open(output_filename, 'wb') as f:
        f.write(response['AudioStream'].read())

def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)

# Main function
async def main():
    
   
    while True:

        print("Intialising..")
        time.sleep(1)
        print("Speak a prompt...")

        with sr.Microphone() as source:
            source.pause_threshold = 1
            recognizer.adjust_for_ambient_noise(source)
                         
            audio = recognizer.listen(source)
            # model = whisper.load_model("tiny") #orig tiny 
            # print("Initialising result")
            # result = ""
            # print(result)
            
            
            try: 
                print("Processing...")
                transcription = recognizer.recognize_google(audio)
                
                #with open("query.wav", "wb") as f:
                    
                #    f.write(audio.get_wav_data()) 

                #result = model.transcribe("query.wav")                
                #phrase = result["text"]
                
                print(f"Querying... You said: '{transcription}'")

                if transcription != "":
                        
                    # Generate the response
                    response = generate_response(transcription)
                    print(f"Chat GPT-3 says: {response}")

                    # Speak the response using text-to-speech
                    speak_text(response)
                        
                else:

                    print("Problem detecting audio, try again...")
                
                
            except Exception as e:
                print("Error transcribing audio: {0}".format(e))
                continue




if __name__ == "__main__":
    asyncio.run(main())

# Wake word logic

# print(f"Waiting for wake words 'ok bing' or 'ok chat'...")
        # while True:
            
        #     audio = recognizer.listen(source)
            
        #     try:
        #         with open("audio.wav", "wb") as f:
        #             print("Processing...")
        #             f.write(audio.get_wav_data())
        #         # Use the preloaded tiny_model
        #         model = whisper.load_model("tiny") #orig tiny
        #         result = model.transcribe("audio.wav")
        #         phrase = result["text"]
        #         print(f"Waking... You said: {phrase}")

        #         wake_word = get_wake_word(phrase)
        #         if wake_word is not None:
        #             print("Wake word recognised")
        #             break
        #         else:
        #             print("Not a wake word. Try again.")                        
        #     except Exception as e:
        #         print("Error transcribing audio: {0}".format(e))
        #         continue

