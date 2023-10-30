#Para copiar de un GGLMv3 a un GGUF:
#python .venv\Lib\site-packages\llama_cpp\convert-llama-ggml-to-gguf.py --eps 1e-5 --input ./llama-2-7b-chat.ggmlv3.q8_0.bin --output ./llama-2-7b-chat.gguf.q8_0.bin 

### Procesado LLM del texto ###
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


template = """Question: answer in the same language, {question}

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
prompt = """
Diga'm la capital de França.
"""

llm(prompt)

# # load the large language model file
# from llama_cpp import Llama
# LLM = Llama(self,model_path="./llama-2-7b-chat.ggmlv3.q8_0.bin")

# # create a text prompt
# prompt = "What are the names of the days of the week?"

# # generate a response (takes several seconds)
# output = LLM(prompt)

# # display the response
# print(output["choices"][0]["text"])