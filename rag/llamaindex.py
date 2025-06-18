#!/usr/bin/env python
# coding: utf-8

# In[26]:


#pip install llama-index
#pip install llama-index-readers-file
#pip install llama-index-embeddings-azure-openai
#pip install llama-index-llms-azure-openai


# In[27]:


import yaml


# In[28]:


def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

config = read_config("C:/Users/Julie-Marie Biffe/Project_RAG/secrets/config.yaml")


# In[29]:


from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

llm = AzureOpenAI(
    model=config["chat"]["azure_deployment"],
    deployment_name=config["chat"]["azure_deployment"],
    api_key=config["chat"]["azure_api_key"],
    azure_endpoint=config["chat"]["azure_endpoint"],
    api_version=config["chat"]["azure_api_version"],
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model=config["embedding"]["azure_deployment"],              # for the moment, same as deployment
    deployment_name=config["embedding"]["azure_deployment"],
    api_key=config["embedding"]["azure_api_key"],
    azure_endpoint=config["embedding"]["azure_endpoint"],
    api_version=config["embedding"]["azure_api_version"],
)


# In[30]:


from llama_index.readers.file import PyMuPDFReader


# In[31]:


loader = PyMuPDFReader()
documents = loader.load(file_path="C:/Users/Julie-Marie Biffe/OneDrive/Documents/mag 3/Mignot/Anomaly_Detection_The_Mathematization_of.pdf")

# documents = SimpleDirectoryReader(
#     input_files=["../../data/paul_graham/paul_graham_essay.txt"]
# ).load_data()


# In[32]:


from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model

index = VectorStoreIndex.from_documents(documents)


# In[33]:


query = "liste les ouvrages mentionnés dans le document"
query_engine = index.as_query_engine()
answer = query_engine.query(query)

print(answer.get_formatted_sources())
print("query was:", query)
print("answer was:", answer)


# In[34]:


from llama_index.core.node_parser import SentenceSplitter

text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)

text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))


# In[35]:


text_chunks


# In[36]:


from llama_index.core.schema import TextNode

nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)


# In[37]:


print(text_chunk)


# In[38]:


for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding


# In[39]:


# from llama_index.vector_stores.postgres import PGVectorStore

# vector_store = PGVectorStore.from_params(
#     database=db_name,
#     host=host,
#     password=password,
#     port=port,
#     user=user,
#     table_name="llama2_paper",
#     embed_dim=384,  # openai embedding dimension
# )

from llama_index.core.vector_stores import SimpleVectorStore

vector_store = SimpleVectorStore()
vector_store.add(nodes)


# In[40]:


len(nodes)


# In[41]:


query_str = "list all books mentioned in the document"
query_embedding = embed_model.get_query_embedding(query_str)
from llama_index.core.vector_stores import VectorStoreQuery

query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=5, mode=query_mode
)

# returns a VectorStoreQueryResult
query_result = vector_store.query(vector_store_query)
if query_result.nodes:
    print(query_result.nodes[0].get_content())
else:
    print('No results')


# In[42]:


query_result


# In[43]:


vector_store.to_dict()


# In[44]:


CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

embedder = AzureOpenAIEmbedding(
    model=config["embedding"]["azure_deployment"],              # for the moment, same as deployment
    deployment_name=config["embedding"]["azure_deployment"],
    api_key=config["embedding"]["azure_api_key"],
    azure_endpoint=config["embedding"]["azure_endpoint"],
    api_version=config["embedding"]["azure_api_version"],
)

vector_store = SimpleVectorStore()


def store_pdf_file(file_path: str, doc_name: str):
    """Store a pdf file in the vector store.

    Args:
        file_path (str): file path to the PDF file
    """
    loader = PyMuPDFReader()
    documents = loader.load(file_path)

    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)
    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata in (3)
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        print(node.id_)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    for node in nodes:
        node_embedding = embedder.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    vector_store.add(nodes)
    return


# In[45]:


store_pdf_file('C:/Users/Julie-Marie Biffe/OneDrive/Documents/mag 3/Mignot/B4LFlaparureguydemaupassant.pdf', 'B4LFlaparureguydemaupassant.pdf')


# In[46]:


def retrieve(question: str):
    """Retrieve documents similar to a question.

    Args:
        question (str): text of the question

    Returns:
        list[TODO]: list of similar documents retrieved from the vector store
    """
    query_embedding = embedder.get_query_embedding(question)

    query_mode = "default"
    # query_mode = "sparse"
    # query_mode = "hybrid"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=5, mode=query_mode
    )

    # returns a VectorStoreQueryResult
    query_result = vector_store.query(vector_store_query)
    return query_result.nodes

    # if query_result.nodes:
    #     print(query_result.nodes[0].get_content())
    # else:
    #     print('No results')


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


def answer_question(question: str) -> str:
    """Answer a question by retrieving similar documents in the store.

    Args:
        question (str): text of the question

    Returns:
        str: text of the answer
    """
    docs = retrieve(question)
    docs_content = "\n\n".join(doc.get_content() for doc in docs)
    print("Question:", question)
    print("------")
    for doc in docs:
        print("Chunk:", doc.id)
        print(doc.page_content)
        print("------")
    messages = build_qa_messages(question, docs_content)
    response = llm.invoke(messages)
    return response.content


# In[47]:


retrieve("comment s'appelle l'amie de madame Loisel") is None


# In[48]:


vector_store.query(VectorStoreQuery(
        query_embedding=embedder.get_query_embedding("comment s'appelle l'amie de madame Loisel"),
        similarity_top_k=5,
        mode=query_mode
    ))


# 

# In[50]:


vector_store.get('f758d76b-7f38-4810-842b-b29b282c35af')


# In[ ]:





# In[ ]:




