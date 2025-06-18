#!/usr/bin/env python
# coding: utf-8

# ## Mise au point de processus de RAG
# 
# ### Installation

# In[ ]:


#pip install langchain langchain-openai langchain_community pymupdf yaml
#pip install langchain==0.3.25 langchain-openai==0.1.7 langchain-community==0.0.36 pymupdf pyyaml



# ### Imports

# In[1]:


import os
import yaml

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import AzureChatOpenAI



#from langchain.chains import RetrievalQA
#from langchain.llms import OpenAI


# ### Configuration

# In[2]:


def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

config = read_config("C:/Users/Julie-Marie Biffe/Project_RAG/secrets/config.yaml")


# ### Initialisation

# In[3]:


from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=config["embedding"]["azure_endpoint"],
    azure_deployment=config["embedding"]["azure_deployment"],
    openai_api_version=config["embedding"]["azure_api_version"],
    api_key=config["embedding"]["azure_api_key"]
)


# In[4]:


from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)


# In[5]:


from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=config["chat"]["azure_endpoint"],
    azure_deployment=config["chat"]["azure_deployment"],
    openai_api_version=config["chat"]["azure_api_version"],
    api_key=config["chat"]["azure_api_key"],
)


# ### Extraction

# In[6]:


file_path = "C:/Users/Julie-Marie Biffe/OneDrive/Documents/mag 3/Mignot/B4LFlaparureguydemaupassant.pdf"
loader = PyMuPDFLoader(file_path)


# In[7]:


docs = loader.load()
docs[0]


# In[8]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)


# In[9]:


len(all_splits)


# ### Indexation

# In[10]:


_ = vector_store.add_documents(documents=all_splits)


# ### Interrogation

# In[11]:


def retrieve(store, question: str):
    retrieved_docs = store.similarity_search(question)
    return retrieved_docs


# In[12]:


# from langchain import hub

# print(hub.pull("rlm/rag-prompt").messages[0].prompt.template)


# In[13]:


def build_qa_messages(question: str, context: str) -> list[str]:
    messages = [
    (
        "system",
        "You are an assistant for question-answering tasks.",
    ),
    (
        "system",
        """Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        {}""".format(context),
    ),
    (  
        "user",
        question
    ),]
    return messages


# In[14]:


question = "comment s'appelle l'amie de madame Loisel"
docs_content = "\n\n".join(doc.page_content for doc in retrieve(vector_store, question))
messages = build_qa_messages(question, docs_content)


# In[15]:


response = llm.invoke(messages)


# In[16]:


print(response.content)


# ### Application à un autre fichier

# In[17]:


file_path = "C:/Users/Julie-Marie Biffe/OneDrive/Documents/mag 3/Mignot/Anomaly_Detection_The_Mathematization_of.pdf"
loader = PyMuPDFLoader(file_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
new_vector_store = InMemoryVectorStore(embeddings)
_ = new_vector_store.add_documents(documents=all_splits)
question = "liste les ouvrages mentionnés dans le document"
docs_content = "\n\n".join(doc.page_content for doc in retrieve(new_vector_store, question))
messages = build_qa_messages(question, docs_content)
response = llm.invoke(messages)
print(response.content)


# In[18]:


question = "liste les auteurs mentionnés dans le document"
docs_content = "\n\n".join(doc.page_content for doc in retrieve(new_vector_store, question))
messages = build_qa_messages(question, docs_content)
response = llm.invoke(messages)
print(response.content)


# In[19]:


question = "liste les organisations mentionnées dans le document"
docs_content = "\n\n".join(doc.page_content for doc in retrieve(new_vector_store, question))
messages = build_qa_messages(question, docs_content)
response = llm.invoke(messages)
print(response.content)


# In[24]:


def get_meta_doc(extract: str) -> str:
    messages = [
    (
        "system",
        "You are a librarian extracting metadata from documents.",
    ),
    (
        "user",
        """Extract from the content the following metadata.
        Answer 'unknown' if you cannot find or generate the information.
        Metadata list:
        - title
        - author
        - source
        - type of content (e.g. scientific paper, litterature, news, etc.)
        - language
        - themes as a list of keywords

        <content>
        {}
        </content>
        """.format(extract),
    ),]
    response = llm.invoke(messages)
    return response.content


# In[25]:


from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

file_path = "C:/Users/Julie-Marie Biffe/OneDrive/Documents/mag 3/Mignot/article_nature.pdf"
loader = PyMuPDFLoader(file_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
extract = '\n\n'.join([split.page_content for split in all_splits[:min(10, len(all_splits))]])


# In[26]:


print(get_meta_doc(extract))


# In[ ]:




