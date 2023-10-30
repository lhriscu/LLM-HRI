### Entrada micro speech-to-text ###

from whisper_mic.whisper_mic import WhisperMic

mic = WhisperMic(model="large")
result = mic.listen()
print(result)

#Model large tarda más que el model base que es solo en inglés.
#Habría que valorar si sale a cuenta un traductor, en algún punto del proceso. 

### Procesado LLM del texto ###

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


template = """Question: answer in the same language and short answers, {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./llama-2-7b-chat.gguf.q8_0.bin",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
prompt = result

resultllm= llm(prompt)

### Text to Speech ###

import pyttsx3
engine = pyttsx3.init() # object creation

""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
#print (rate)                        #printing current voice rate
engine.setProperty('rate', 125)     # setting up new voice rate


"""VOLUME"""
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
#print (volume)                          #printing current volume level
engine.setProperty('volume',1.00)    # setting up volume level  between 0 and 1

"""VOICE"""
#voices = engine.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
#engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

engine.say(resultllm)
#engine.say('My current speaking rate is ' + str(rate))
engine.runAndWait()
engine.stop()

"""Saving Voice to a file"""
# On linux make sure that 'espeak' and 'ffmpeg' are installed
#engine.save_to_file('Hello World', 'test.mp3')
#engine.runAndWait()