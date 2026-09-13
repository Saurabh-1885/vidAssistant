from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os


# ---------------------------------------------------------
# MISTRAL LLM
# ---------------------------------------------------------

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2
    )


# ---------------------------------------------------------
# SPLIT TRANSCRIPT
# ---------------------------------------------------------

def split_transcript(transcript: str) -> list:
    """
    Split the transcript into larger chunks.

    This does NOT modify the original transcript.
    It is only used for summarization.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=15000,
        chunk_overlap=300
    )

    return splitter.split_text(transcript)


# ---------------------------------------------------------
# SUMMARIZE
# ---------------------------------------------------------

def summarize(transcript: str) -> str:

    if not transcript or not transcript.strip():
        return "No transcript available."

    llm = get_llm()

    # -----------------------------------------------------
    # SHORT TRANSCRIPT
    # -----------------------------------------------------
    # One API call instead of splitting it into chunks.

    if len(transcript) <= 15000:

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert meeting summarizer.

Summarize the following meeting transcript.

Include:
- Main topics discussed
- Important points
- Decisions
- Action items
- Deadlines or dates
- Important conclusions

Rules:
- Use ONLY information from the transcript.
- Do not invent information.
- Do not change the meaning.
- Preserve names, dates and important numbers.
- Keep the summary concise.
- Use clear bullet points.
"""
                ),
                (
                    "human",
                    "{text}"
                )
            ]
        )

        chain = prompt | llm | StrOutputParser()

        return chain.invoke({
            "text": transcript
        })

    # -----------------------------------------------------
    # LONG TRANSCRIPT
    # -----------------------------------------------------

    chunks = split_transcript(transcript)

    print(f"Transcript length: {len(transcript)} characters")
    print(f"Number of chunks: {len(chunks)}")

    chunk_summaries = []

    for i, chunk in enumerate(chunks, start=1):

        print(f"Summarizing chunk {i}/{len(chunks)}...")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert meeting summarizer.

Summarize this portion of a meeting transcript.

Preserve:
- Important topics
- Decisions
- Action items
- Names
- Dates
- Deadlines
- Numbers
- Important conclusions

Rules:
- Use ONLY information present in this transcript.
- Do not invent information.
- Do not change the meaning.
- Do not omit important decisions or action items.
- Be concise.
"""
                ),
                (
                    "human",
                    "{text}"
                )
            ]
        )

        chain = prompt | llm | StrOutputParser()

        summary = chain.invoke({
            "text": chunk
        })

        chunk_summaries.append(summary)

    # -----------------------------------------------------
    # COMBINE SUMMARIES
    # -----------------------------------------------------

    combined = "\n\n".join(chunk_summaries)

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert meeting summarizer.

Combine the partial summaries below into one final
professional meeting summary.

Use these sections:

### Overview
### Key Discussion Points
### Decisions
### Action Items
### Important Dates / Deadlines

Rules:
- Use ONLY information from the partial summaries.
- Do not invent information.
- Do not change the meaning.
- Do not repeat information.
- Preserve names, dates, deadlines and numbers.
- Keep the final summary concise.
- Use bullet points.
"""
            ),
            (
                "human",
                "{text}"
            )
        ]
    )

    final_chain = final_prompt | llm | StrOutputParser()

    print("Creating final summary...")

    return final_chain.invoke({
        "text": combined
    })


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

def generate_title(transcript: str) -> str:
    """
    Generate a title without using the Mistral API.
    This saves one API request.
    """

    if not transcript or not transcript.strip():
        return "AI Meeting Analysis"

    # Take the beginning of the transcript
    text = transcript.strip()

    # Get first sentence
    first_sentence = text.split(".")[0].strip()

    if not first_sentence:
        return "AI Meeting Analysis"

    # Keep title short
    if len(first_sentence) > 60:
        first_sentence = first_sentence[:60].rsplit(" ", 1)[0]

    return first_sentence