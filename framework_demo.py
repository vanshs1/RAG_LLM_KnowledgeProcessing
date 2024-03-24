#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example code to be included as supplementary material to the following article: 
"A Reliable Knowledge Processing Framework for Combustion Science using Foundation Models".

This is a demonstration intended to provide a working example of the knowledge processing workflow. 
The sample document data is provided in this code.
For application on other datasets, the requirement is to replace the document data source. 

Dependencies: 
    - Python package list provided: package.list
    - Quantized model can be downloaded from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
      For this you can use - llama-2-7b-chat.ggmlv3.q8_0

Running the code: 
    - assuming above dependencies are configured, and create_database.py code has generated a database,
    "python framework_demo.py" will run the code for Q&A. 

Authors: 
    - Vansh Sharma, Venkat Raman

Affiliation: 
    - APCL Group 
    - Department of Aerospace Engineering, University of Michigan, Ann Arbor
"""

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import chromadb

#%% Phase 1 - Load DB
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5')

### L2 distance norm - Quite similar to cosine for scientific case
# persist_directory = "/dbs_fullScale/"

# db_name = "DB_fullScale_cSize_%d_cOver_%d" %(2000, 300)

### Cosine based distance
persist_directory = "./dbs_fullScale_cosine/"
db_name = "DB_cosine_cSize_%d_cOver_%d" % (700, 200)

print('\nLoading database from:',persist_directory)
_client_settings = chromadb.PersistentClient(path=(persist_directory+db_name)) 

vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embeddings_model, 
                  client=_client_settings,
                  collection_name=db_name)

print('\nDatabase loaded...')

#%% Phase 2 - Setup the LLM Model for Q&A
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

model_path = "llama-2-7b-chat.ggmlv3.q8_0.bin"

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 0 # Change this value based on your model and your GPU VRAM pool.
n_batch = 512 # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
    model_path=model_path, n_ctx = 100000, max_tokens = -1,
    # n_gpu_layers=n_gpu_layers, n_batch=n_batch,
    callback_manager=callback_manager, 
    verbose=True,
)


#%% Phase 3 - Setup LLM Chain with Prompt
# Build prompt
from langchain.prompts import PromptTemplate

### This template is a demo template - change it as per your topic  
template = """Use the following pieces of context to answer the question at the end. You are a subject matter expert in Oblique Detonation waves and their numerical analysis.
Always say "thanks for asking!" at the end of the answer.
Context: {context}
Question: {question}
Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(template=template, input_variables=["context" ,"question"],)

question = "what is oblique detonation wave?" 

extra = " Reply with minimum 500 words and provide give a detailed list of research papers for this topic. If you don't know the answer, just say that you don't know, don't try to make up an answer. If you dont know the full research paper name, do not try to make up a research article name"

retriever = vectordb.as_retriever(search_kwargs={"k": 4}) ## Pick top 4 results from the search
unique_docs = retriever.get_relevant_documents(question)
question = question + extra 

# Following is the QA system ----------------------------------------------
qa = RetrievalQA.from_chain_type(llm=llm, 
                                  retriever=retriever,
                                  return_source_documents=True,
                                  chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

result = qa({"query": question})
result["result"]




