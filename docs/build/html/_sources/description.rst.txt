myrag app demo 
================

.. note::
    Before we start , we need to make sure that all the depencies from the requirements.txt are installed.

    .. code-block:: bash

        pip install -r requirements.txt

This is a simple demo of the myrag app. The app is a simple web application that allows users to create, view, update and delete notes. The app is built using the Flask web framework and the SQLAlchemy ORM.

Importing libraries 
-------------------
.. code-block:: python

    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_community import embeddings
    from langchain_community.chat_models import ChatOllama
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_nomic.embeddings import NomicEmbeddings

    from langchain.schema import Document
    import streamlit as st
    import os 
    import PyPDF2


Reading the pdf file
--------------------
.. code-block:: python
    
    def read_pdf(file):
        pdfReader = PyPDF2.PdfReader(file)
        all_page_text = ""
        for page in pdfReader.pages:
            all_page_text += page.extract_text() + "\n"
        return all_page_text


Retreive & Answer for a pdf file
------------------------------------------------------------
.. code-block:: python

    def retriever(doc, question):
        model_local = ChatOllama(model="mistral")
        doc = Document(page_content=doc)
        doc = [doc]
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=0)
        doc_splits = text_splitter.split_documents(doc)

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5"),
        )
        retriever = vectorstore.as_retriever(k=2)
        after_rag_template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
        after_rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | after_rag_prompt
            | model_local
            | StrOutputParser()
        )

        return after_rag_chain.invoke(question)

As you can see here we using an embedding model called NomicEmbeddings. This model is used to convert the text into embeddings. The embeddings are then used to find the most similar text to the question. The most similar text is then used to answer the question.
Also NomicEmbeddings is a model that is open source and can be found in the langchain_community package. And it also well ranked in the leaderboard of the embedding models.

.. note::
    In order to use the NomicEmbeddings model, you need to pass a Nomic_API_KEY as an environment variable. You can get the API key by signing up on the Nomic website.

Retreive & Answer for a web page
---------------------------------

.. code-block:: python

    
    def retriever_with_links(question, links):
        model_local = ChatOllama(model="mistral")
        docs = [WebBaseLoader(url).load() for url in links]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(docs_list)

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5"),
        )
        retriever = vectorstore.as_retriever(k=1)
        after_rag_template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
        after_rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | after_rag_prompt
            | model_local
            | StrOutputParser()
        )
        print("done")
        return after_rag_chain.invoke(question)



streamlit app
--------------------------------------------
.. code-block:: python

    st.title("RAG with retriever")

    # switch button to choose between pdf and website
    switch = st.radio("Choose between pdf or website", ["pdf", "website"])
    if switch == "pdf":
        file = st.file_uploader("Upload a pdf file", type=["pdf"])
        if file:
            st.write("File uploaded")
            text = read_pdf(file)
            question = st.text_input("Ask a question")
            if st.button("Ask"):
                answer = retriever(text, question)
                st.write(answer)
    else:
        links = st.text_input("Enter the links separated by commas")
        links = links.split(",")
        question = st.text_input("Ask a question")
        if st.button("Ask"):
            answer = retriever_with_links(question, links)
            st.write(answer)

