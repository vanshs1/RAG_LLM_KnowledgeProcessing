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

Running the code: 
    - assuming above dependencies are configured, "python create_database.py" will run the code. 
    NOTE - It is important to run create_database.py prior to running framework_demo.py code.

Authors: 
    - Vansh Sharma, Venkat Raman

Affiliation: 
    - APCL Group 
    - Department of Aerospace Engineering, University of Michigan, Ann Arbor
"""

import subprocess
import os
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb

local = os.getcwd()  ## Get local dir
os.chdir(local)      ## shift the work dir to local dir
print('\nWork Directory: {}'.format(local))

#%% Phase 1 - Load DB
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-base-en-v1.5')

#%% Phase 1 - Load documents
path_docs = './docs_ODW/'

print('\nDocuments loading from:',path_docs)
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(path_docs, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader, 
                          loader_kwargs=text_loader_kwargs, show_progress=True,
                          use_multithreading=True)
docs_data = loader.load()  
print('\nDocuments loaded...')

#%% Phase 2 - Split the text
from langchain.text_splitter import RecursiveCharacterTextSplitter
persist_directory = "./dbs_fullScale_cosine/"

## User input ::
arr_chunk_size = [700] #Chunk size 
arr_chunk_overlap = [200] #Chunk overlap

for i in range(len(arr_chunk_size)):
    for j in range(len(arr_chunk_overlap)):
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = arr_chunk_size[i], 
                                                        chunk_overlap = arr_chunk_overlap[j], 
                                                        separators=[" ", ",", "\n", ". "])
        data_splits = text_splitter.split_documents(docs_data)
            
        print('\nDocuments split into chunks...')
        
        #%% Phase 2 - Split the text
        print('\nInitializing Chroma Database...')
        db_name = "DB_cosine_cSize_%d_cOver_%d" %(arr_chunk_size[i], arr_chunk_overlap[j])
        
        p2_2 = subprocess.run('mkdir  %s/*'%(persist_directory+db_name), shell=True)
        _client_settings = chromadb.PersistentClient(path=(persist_directory+db_name))
        
        vectordb = Chroma.from_documents(documents = data_splits, 
                                    embedding = embeddings_model, 
                                    client = _client_settings,
                                    collection_name = db_name,
                                    collection_metadata={"hnsw:space": "cosine"})
        
        print('Completed Chroma Database: ', db_name)
        del vectordb, text_splitter, data_splits
        


            






