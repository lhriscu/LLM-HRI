from google.cloud import speech
import numpy as np
import speech_recognition as sr

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
 
language= "es-ES"

### Listening ###

# Obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    if language == "es-ES":
        print(f"Has escogido {language} como idioma.")
        print("Hola! En qué puedo ayudarte?")
        audio = r.listen(source)
    elif language == "en-US":
        print(f"You have chosen {language} language.")
        print("Hello! How can I help you?")
        audio = r.listen(source)

# Transcribe audio
try:
    result = r.recognize_google_cloud(audio, credentials_json='keygoogle.json', language=language)
except sr.UnknownValueError:
    print("Google Cloud Speech could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Cloud Speech service; {0}".format(e))

### LLM ###

# Parametritzar això:
prompt_template = """<s>[INST] <<SYS>>
{{ You are a helpful AI Assistant that receives a question, 
translates it to English to process it better and then answers 
the question in the same original language.}}<<SYS>>
###

Previous Conversation:
'''
{history}
'''

{{{input}}}[/INST]

"""
prompt = PromptTemplate(template=prompt_template, input_variables=['input', 'history']) # Provar a dir-li idioma com a input.
# Callbacks support token-wise streaming
#callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="./llama-2-7b-chat.gguf.q8_0.bin",
    temperature=0.75,
    max_tokens=200,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose required to pass to the callback manager
)

prompt = result
resultllm = llm(prompt)

# from llama_cpp import Llama
# llm = Llama(model_path="./llama-2-7b-chat.gguf.q8_0.bin")
# output = llm(f"Q: {result} ? A: ", max_tokens=32, stop=["Q:", "\n"], echo=False)
# print(output)

