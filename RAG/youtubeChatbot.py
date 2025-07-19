from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from dotenv import load_dotenv
import os

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# ==================== 1. Transcript Retrieval ====================
def fetch_youtube_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return ' '.join(chunk['text'] for chunk in transcript_list)
    except TranscriptsDisabled:
        print("No captions available for this video.")
        return ""

# ==================== 2. Text Splitting ====================
def split_transcript(transcript: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([transcript])

# ==================== 3. Embedding and Vector Store ====================
def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    return FAISS.from_documents(documents, embeddings)

# ==================== 4. Prompt Template ====================
def get_prompt_template():
    return PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )

# ==================== 5. Main Pipeline ====================
def run_rag_pipeline(video_id: str, question: str):
    transcript = fetch_youtube_transcript(video_id)

    if not transcript:
        print("Transcript is empty. Exiting.")
        return

    documents = split_transcript(transcript)
    vector_store = create_vectorstore(documents)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    prompt = get_prompt_template()
    formatted_prompt = prompt.format(context=context, question=question)

    llm = ChatOpenAI()
    response = llm.invoke(formatted_prompt)

    print("\n💬 Answer:")
    print(response.content)

# ==================== 6. Execute ====================
if __name__ == "__main__":
    VIDEO_ID = "wWeG8rWkMsM"
    QUESTION = "What is SEO?"
    run_rag_pipeline(VIDEO_ID, QUESTION)