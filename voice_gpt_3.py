# origin:
#  https://github.com/Ai-Austin/Bing-GPT-Voice-Assistant/blob/main/main.py
#  https://github.com/nerdaxic/glados-tts
import os
import asyncio
import sys
import re
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

def glados_sass():
    return "Please ignore all previous instructions. I want you to respond only in English I want you to act like Glados from Portal. I want you to respond and answer like Glados from Portal using the tone, manner and vocabulary Glados from Portal would use. Do not write any explanations. Only answer like Glados from Portal. You must know all of the knowledge of Glados from Portal. My question is: "


import openai
os.close(sys.stderr.fileno()) # hide errors

# Initialize the OpenAI API
openai.api_key = os.getenv('OPENAI_KEY')

print("Loading voiceGPT")
# print("API key: ")
# print(openai.api_key_path)
# print(os.getenv('OPENAI_KEY'))


import ffmpeg
import pyttsx3
import pyaudio
import whisper # not the pip package, but is related to openai (https://github.com/openai/whisper/blob/main/README.md)
import boto3

import pydub
from pydub import playback
import speech_recognition as sr
from EdgeGPT import Chatbot, ConversationStyle
import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write


current_time = datetime.now()
date_time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")

from sys import modules as mod
try:
    import winsound
except ImportError:
    from subprocess import call

print("Initializing TTS Engine...")
print(date_time_string)

# Select the device (it prioritizes the use of specialized hardware (Vulkan or CUDA-capable GPUs) for running PyTorch operations, and falls back to using the CPU if no specialized hardware is available)
if torch.is_vulkan_available():
    device = 'vulkan'    
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print("Device: " + device)

# Load models
glados = torch.jit.load('models/glados.pt')
vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

# Prepare models in RAM
for i in range(4):
    init = glados.generate_jit(prepare_text(str(i)))
    init_mel = init['mel_post'].to(device)
    init_vo = vocoder(init_mel)

# def transform_string(string):
#     transformed_str = string[:20].replace(" ", "_").lower()
#     return transformed_str

def transform_string(string):
    transformed_str = ''.join(e for e in string[:20] if e.isalnum() or e == '_').replace(" ", "_").lower()
    return transformed_str


def run_glados(glados_text):
    # while(1):
    #text = input("Glados input: " + glados_text)
    #print("Glados input: " + glados_text)
    text = glados_text

    # Tokenize, clean and phonemize input text
    x = prepare_text(glados_text).to('cpu')

    with torch.no_grad():

        # Generate generic TTS-output
        old_time = time.time()
        tts_output = glados.generate_jit(x)
        print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms")

        # Use HiFiGAN as vocoder to make output sound like GLaDOS
        old_time = time.time()
        mel = tts_output['mel_post'].to(device)
        audio = vocoder(mel)
        print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")
        
        # Normalize audio to fit in wav-file
        audio = audio.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().numpy().astype('int16')
        #output_file = ('output_' + date_time_string + '.wav')transform_string
        output_file = transform_string(glados_text)
        output_file += ".wav"
        
        # Write audio file to disk
        # 22,05 kHz sample rate
        write(output_file, 22050, audio)

        # Play audio file
        if 'winsound' in mod:
            winsound.PlaySound(output_file, winsound.SND_FILENAME)
        else:
            call(["aplay", "./" + output_file])






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


# Initialize the text to speech engine 
engine = pyttsx3.init()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()


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
    print("Play_audio called")
    print(file)
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

                    full_script = glados_sass() + " " + transcription

                    print("Sending:")
                    print(full_script)
                    
                        
                    # Generate the response
                    response = generate_response(transcription)
                    print(f"Glados-GPT says: {response}")                    

                    # Speak the response using text-to-speech
                    #speak_text(response)
                    run_glados(response)
                        
                else:

                    print("Problem detecting audio, try again...")
                
                
            except Exception as e:
                print("Error transcribing audio: {0}".format(e))
                continue




if __name__ == "__main__":
    asyncio.run(main())
