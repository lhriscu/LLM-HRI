from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings

# Funciona però va excessivament lent, provar amb documents PDF i bona cobertura. 
#loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
#loader = WebBaseLoader("https://www.bbc.com/news/business-67353177") #Exemple 1
loader = WebBaseLoader("https://www.upc.edu/en/masters/statistics-and-operations-research")
data = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

#vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./llama-2-7b-chat.gguf.q8_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048, #window context
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)


# Prompt
prompt = PromptTemplate.from_template(
    #"Summarize the main themes in these retrieved docs: {docs}"
    "Given the information in retrieved docs: {docs}, return only the answer to the question."
)

# Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Run
#question = "What are the approaches to Task Decomposition?"
#question = "What is the impact of interest rates for China business investors?" #Exemple 1

question = "What are the tuition fees of Master's degree in Statistics and Operations Research?" #Exemple 2
docs = vectorstore.similarity_search(question)
result = llm_chain(docs)

# Output
result["text"]

