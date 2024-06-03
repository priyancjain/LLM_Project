import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain import PromptTemplate
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import sys

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    return all_splits

def create_vectorstore(documents):
    with SuppressStdout():
        vectorstore = Chroma.from_documents(documents=documents, embedding=GPT4AllEmbeddings())
    return vectorstore

def prepare_prompt_template():
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    return PromptTemplate(input_variables=["context", "question"], template=template)

def create_qa_chain(vectorstore, prompt_template):
    llm = Ollama(model="llama2:7b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt_template},
    )
    return qa_chain

def main():
    st.title("PDF Question Answering System")

    # Initialize session state variables
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'queries' not in st.session_state:
        st.session_state.queries = []
    if 'answers' not in st.session_state:
        st.session_state.answers = []

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File uploaded successfully!")

        if st.button("Process PDF"):
            documents = load_and_split_pdf("uploaded_file.pdf")
            vectorstore = create_vectorstore(documents)
            prompt_template = prepare_prompt_template()
            qa_chain = create_qa_chain(vectorstore, prompt_template)
            st.session_state.qa_chain = qa_chain
            st.session_state.queries = []
            st.session_state.answers = []
            st.success("PDF processed and vector store created!")

    if st.session_state.qa_chain:
        query = st.text_input("Enter your query:")
        if st.button("Submit Query"):
            if query:
                result = st.session_state.qa_chain({"query": query})
                st.session_state.queries.append(query)
                st.session_state.answers.append(result['result'])

    if st.session_state.queries and st.session_state.answers:
        st.subheader("Previous Queries and Answers:")
        for q, a in zip(st.session_state.queries, st.session_state.answers):
            st.write(f"**Query:** {q}")
            st.write(f"**Answer:** {a}")
            st.write("---")

if __name__ == "__main__":
    main()
