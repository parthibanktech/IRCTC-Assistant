import streamlit as st
import pypdf
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# --- Page config ---
st.set_page_config(
    page_title="🚆 Indian Railways Assistant",
    page_icon="🚆",
    layout="wide"
)

# --- Custom CSS for modern UI ---
st.markdown("""
<style>
/* Background */
body {
    background: linear-gradient(135deg, #e0f7fa, #fffde7);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Sidebar */
.stSidebar {
    background-color: #ffffff;
    padding: 30px;
    border-right: 1px solid #ddd;
    box-shadow: 2px 0 5px rgba(0,0,0,0.05);
}
.stSidebar h3 {
    color: #d32f2f;
}

/* Header */
h1 {
    font-size: 42px;
    font-weight: 700;
    color: #d32f2f;
    text-align: center;
    margin-bottom: 0px;
}
p.header-desc {
    font-size: 18px;
    color: #555;
    text-align: center;
    margin-top: 5px;
    margin-bottom: 30px;
}

/* Chat bubbles */
.stChatMessage {
    border-radius: 25px;
    padding: 16px 24px;
    margin-bottom: 16px;
    max-width: 75%;
    font-size: 16px;
    line-height: 1.6;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    word-wrap: break-word;
}
.stChatMessageUser {
    background: linear-gradient(135deg, #bbdefb, #90caf9);
    align-self: flex-end;
    color: #0d47a1;
}
.stChatMessageAssistant {
    background: linear-gradient(135deg, #c8e6c9, #a5d6a7);
    align-self: flex-start;
    color: #1b5e20;
}

/* Chat input */
.css-1f2kx4t {
    border-radius: 20px;
    padding: 14px;
    font-size: 16px;
}

/* Timestamp */
.timestamp {
    font-size: 12px;
    color: gray;
    margin-top: 3px;
    display: block;
    text-align: right;
}

/* Source links */
.source-text {
    font-size: 13px;
    color: #555;
    margin-top: 8px;
    font-style: italic;
}

/* Sidebar links & logo */
.sidebar-logo {
    display: block;
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar content ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/45/IRCTC_Logo.svg", width=180)
    st.markdown("### 🚆 Welcome Indian Railways Assistant")
    with st.expander("ℹ️ About this Assistant", expanded=True):
        st.markdown("""
        Built with ❤️ by **Parthiban K**  
        Ask me about:
        - Ticket cancellation rules  
        - Tatkal refunds  
        - Train cancelled & refund  
        - Waitlist rules  
        """)
    st.markdown("---")

# --- Header ---
st.markdown("<h1>🚆 Indian Railways Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-desc'>Your AI assistant for IRCTC cancellation & refund rules. Ask your question below!</p>", unsafe_allow_html=True)

# --- OpenAI Key ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --- Embeddings ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# --- Load PDFs ---
def load_docs():
    docs = []
    pdf_paths = [
        "data/CancellationRulesforIRCTCTrain.pdf",
        "data/ETicketCancellationRefund Rules.pdf"
    ]
    for path in pdf_paths:
        reader = pypdf.PdfReader(path)
        pages = [page.extract_text() for page in reader.pages if page.extract_text()]
        full_text = "\n".join(pages)
        docs.append(Document(page_content=full_text, metadata={"source": path}))
    return docs

# --- Chunk documents ---
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(docs)

# --- Vectorstore & Retriever ---
docs = load_docs()
chunks = chunk_docs(docs)
vectorstore = Chroma.from_documents(chunks, embeddings)
bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 3
embedding_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever = EnsembleRetriever(retrievers=[bm25, embedding_retriever], weights=[0.5, 0.5])

# --- Memory & LLM ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, return_source_documents=True, output_key="answer"
)

# --- Chat session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat history ---
for msg in st.session_state.messages:
    timestamp = msg.get("time", datetime.now().strftime("%H:%M"))
    if msg["role"] == "user":
        st.chat_message("user", avatar="🧑").markdown(
            f"**You:** {msg['content']}  \n<span class='timestamp'>{timestamp}</span>",
            unsafe_allow_html=True
        )
    else:
        st.chat_message("assistant", avatar="🤖").markdown(
            f"**Assistant:** {msg['content']}  \n<span class='timestamp'>{timestamp}</span>",
            unsafe_allow_html=True
        )

# --- Input new question ---
if prompt := st.chat_input("Ask about cancellation, refund, Tatkal..."):
    timestamp = datetime.now().strftime("%H:%M")
    st.chat_message("user", avatar="🧑").markdown(
        f"**You:** {prompt}  \n<span class='timestamp'>{timestamp}</span>", unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "user", "content": prompt, "time": timestamp})

    # Run LLM
    resp = qa_chain({"question": prompt})
    answer = resp["answer"]
    sources = [doc.metadata["source"] for doc in resp.get("source_documents", [])]

    timestamp = datetime.now().strftime("%H:%M")
    st.chat_message("assistant", avatar="🤖").markdown(
        f"**Assistant:** {answer}  \n<span class='timestamp'>{timestamp}</span>", unsafe_allow_html=True
    )

    if sources:
        st.markdown(f"<span class='source-text'>Sources: {', '.join(set(sources))}</span>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "time": timestamp})
