#!/usr/bin/env python
# coding: utf-8

# In[1]:


# in_memory_vectorstore.py
from langchain_core.vectorstores import VectorStore

class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self._docs = []
        self._embeddings = []

    def add_documents(self, documents, embeddings=None):
        self._docs.extend(documents)
        if embeddings:
            self._embeddings.extend(embeddings)

    def similarity_search(self, query, k=4, **kwargs):
        # Cette version ne fait rien de réel, mais ne plantera pas
        return self._docs[:k]


# In[ ]:




