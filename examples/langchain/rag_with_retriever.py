"""
LangChain RAG (Retrieval-Augmented Generation) example with Chaukas instrumentation.

This example demonstrates LangChain retriever usage. It captures:
- DATA_ACCESS events (retriever calls)
- MODEL_INVOCATION events
- SESSION_START/END events
- All standard chain events
"""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import chaukas

# Enable Chaukas instrumentation (one-line setup!)
chaukas.enable_chaukas()


def main():
    """Run a RAG chain with Chaukas instrumentation."""

    # Create sample documents
    documents = [
        Document(page_content="LangChain is a framework for developing applications powered by language models.", metadata={"source": "doc1"}),
        Document(page_content="Chaukas SDK provides one-line observability for agent building frameworks.", metadata={"source": "doc2"}),
        Document(page_content="RAG (Retrieval-Augmented Generation) combines retrieval with generation for better responses.", metadata={"source": "doc3"}),
        Document(page_content="Vector databases store embeddings for semantic search capabilities.", metadata={"source": "doc4"}),
    ]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = text_splitter.split_documents(documents)

    # Create vector store
    print("Creating vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Create RAG prompt
    template = """Answer the question based only on the following context:

{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Create LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Create RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Run the chain - Chaukas automatically tracks everything!
    print("\nRunning RAG chain...")
    result = rag_chain.invoke("What is Chaukas SDK?")

    print("\nResult:", result)
    print("\n✅ Events captured by Chaukas")


if __name__ == "__main__":
    import time
    main()

    # Give async operations time to complete
    time.sleep(0.5)

    # Explicitly disable Chaukas to flush events to file
    chaukas.disable_chaukas()

    print("✅ Events written to file")
