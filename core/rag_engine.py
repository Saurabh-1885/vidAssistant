import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Import functions used to create/load the ChromaDB vector store
# and retrieve relevant transcript chunks
from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)



# Create and configure the Mistral LLM
def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",

        # Read the Mistral API key from the .env environment variable
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),

        # Controls how creative/random the model's responses are
        # Lower value = more consistent and focused answers
        temperature=0.3,
    )



# Convert retrieved documents into plain text
def format_docs(docs):

    # Combine the content of all retrieved transcript chunks
    # into one string, separated by blank lines
    return "\n\n".join([doc.page_content for doc in docs])



# Build a RAG chain using a new meeting transcript
def build_rag_chain(transcript: str):

    # Convert the transcript into embeddings and store them
    # in the ChromaDB vector store
    vector_store = build_vector_store(transcript)

    # Create a retriever that searches for the most relevant
    # transcript chunks. k=4 means retrieve 4 relevant chunks.
    retriever = get_retriever(vector_store, k=4)

    # Initialize the Mistral language model
    llm = get_llm()

    # Create the prompt that will be sent to the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
            ),

            # The user's actual question
            ("human", "{question}"),
        ]
    )

    
    # Full LCEL RAG pipeline
    rag_chain = (

        # "context":
        # Retrieve relevant transcript chunks and convert
        # them into a single text string
        {
            "context": retriever | RunnableLambda(format_docs),

            # "question":
            # Pass the user's question directly through the chain
            "question": RunnablePassthrough()
        }

        # Send context + question into the prompt
        | prompt

        # Send the formatted prompt to Mistral
        | llm

        # Convert the LLM response into a normal string
        | StrOutputParser()
    )

    # Return the completed RAG chain so it can be used for questions
    return rag_chain


# Load an existing RAG chain/vector store
def load_rag_chain():

    # Load the previously created ChromaDB vector store
    # instead of creating a new one
    vector_store = load_vector_store()

    # Create a retriever from the existing vector store
    retriver = get_retriever()

    # Initialize the Mistral language model
    llm = get_llm()

    # Create the same prompt used by the RAG pipeline
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
            ),

            # User's question
            ("human", "{question}"),
        ]
    )

    # Build the RAG pipeline using the existing vector store
    rag_chain = (
        {
            # Retrieve relevant chunks and format them as text
            "context": retriver | RunnableLambda(format_docs),

            # Pass the user's question through
            "question": RunnablePassthrough(),
        }

        # Create the final prompt
        | prompt

        # Send prompt to Mistral
        | llm

        # Convert response to a string
        | StrOutputParser()
    )

    # Return the ready-to-use RAG chain
    return rag_chain


# Ask a question using the RAG chain
def ask_question(rag_chain, question: str):

    # Display the question in the terminal
    print(f"Question : {question}")

    # Send the question through the complete RAG pipeline
    # The pipeline retrieves relevant transcript context,
    # adds it to the prompt, and asks Mistral for an answer.
    answer = rag_chain.invoke(question)

    # Display the generated answer
    print(f"answer :{answer}")

    # Return the answer so other parts of the application
    # can use it
    return answer