import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import validators

def url_process(urls, question):
    model = Ollama(model="llama3")
    retriever = ""

    if urls:
        url_list = urls.split("\n")
        valid_urls = [url for url in url_list if validators.url(url)]

        if valid_urls:
            doc_list = []
            for url in valid_urls:
                doc = WebBaseLoader(url).load()
                doc_list.extend(doc)

            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
            doc_splitter = text_splitter.split_documents(doc_list)

            vector_store = Chroma.from_documents(doc_splitter, collection_name="rag",
                                                 embedding=OllamaEmbeddings(model="snowflake-arctic-embed"))
            retriever = vector_store.as_retriever()

    if retriever == "":
        prompt_template = """Answer the question:
        Question: {question}
        """
        context_question = {"question": RunnablePassthrough()}
    else:
        prompt_template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        context_question = {"context": retriever, "question": RunnablePassthrough()}

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (context_question | prompt | model | StrOutputParser())
    return chain.invoke(question)

st.title("Urly")
st.write("Query the web")
urls = st.text_area("Enter URLs:",height=150)
question = st.text_input("Question")

col1,col2 = st.columns(2)

def doc_btn():
    if st.button("Query url"):
        with st.spinner("Loading///"):
            answer_doc = url_process(urls,question)
            return answer_doc
    return None

def model_btn():
    if st.button("Query Model"):
        with st.spinner("Loading///"):
            answer_model = url_process("",question)
            return answer_model
    return None

with col1:
    answer_doc = doc_btn()
with col2:
    answer_model = model_btn()

if answer_doc :
    st.text_area("Answer Document", value=answer_doc, height=300,disabled=True)

if answer_model:
    st.text_area("Answer Model", value = answer_model , height=300,disabled=True)

