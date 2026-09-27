import os
import re
import json
import sqlite3
from datetime import datetime

import streamlit as st
import numpy as np
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Advanced Agentic RAG Lab",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CONSTANTS
# =========================================================

MODEL_NAME = "openai/gpt-oss-120b"
DATABASE_FILE = "agent_memory.db"


# =========================================================
# HEADER
# =========================================================

st.title("🤖 Advanced Agentic RAG Lab")

st.caption(
    "RAG + Tool Calling + Agent + Persistent Memory + SQLite"
)


# =========================================================
# GROQ API
# =========================================================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. "
        "Add it to Streamlit Secrets."
    )
    st.stop()


client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# SESSION STATE
# =========================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# DATABASE
# =========================================================

def initialize_database():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


initialize_database()


# =========================================================
# SAVE MEMORY
# =========================================================

def save_memory(memory):

    memory = memory.strip()

    if not memory:
        return "Memory cannot be empty."

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO memories
        (memory, created_at)
        VALUES (?, ?)
        """,
        (
            memory,
            datetime.now().isoformat(
                timespec="seconds"
            )
        )
    )

    connection.commit()
    connection.close()

    return "Memory saved successfully."


# =========================================================
# GET ALL MEMORIES
# =========================================================

def get_memories(limit=20):

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id, memory, created_at
        FROM memories
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# =========================================================
# SEARCH MEMORY
# =========================================================

def search_memory(query, limit=10):

    query = query.strip()

    if not query:
        return []

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    # Convert the user's query into useful words.
    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query.lower()
    )

    words = [
        word
        for word in words
        if len(word) > 2
    ]

    if not words:
        connection.close()
        return []

    conditions = []
    parameters = []

    for word in words:

        conditions.append(
            "LOWER(memory) LIKE ?"
        )

        parameters.append(
            f"%{word}%"
        )

    sql = f"""
        SELECT id, memory, created_at
        FROM memories
        WHERE {" OR ".join(conditions)}
        ORDER BY id DESC
        LIMIT ?
    """

    parameters.append(limit)

    cursor.execute(
        sql,
        parameters
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# =========================================================
# DELETE ALL MEMORY
# =========================================================

def clear_all_memories():

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM memories"
    )

    connection.commit()
    connection.close()


# =========================================================
# DOCUMENT EXTRACTION
# =========================================================

def extract_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def extract_docx(file):

    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            text += paragraph.text + "\n"

    return text


def extract_txt(file):

    return file.read().decode(
        "utf-8"
    )


# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(
    text,
    chunk_size=700,
    overlap=100
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        start += (
            chunk_size - overlap
        )

    return chunks


# =========================================================
# BUILD VECTOR DATABASE
# =========================================================

def build_vector_database(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings.astype("float32")
    )

    return index


# =========================================================
# RAG SEARCH
# =========================================================

def rag_search(
    query,
    top_k=4
):

    if (
        st.session_state.index is None
        or not st.session_state.chunks
    ):
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = (
        st.session_state.index.search(
            query_embedding.astype(
                "float32"
            ),
            top_k
        )
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        results.append(
            {
                "chunk": st.session_state.chunks[idx],
                "score": float(score),
                "index": int(idx)
            }
        )

    return results


# =========================================================
# CALCULATOR
# =========================================================

def calculator(expression):

    try:

        expression = expression.strip()

        if not re.fullmatch(
            r"[0-9+\-*/().%\s]+",
            expression
        ):
            return (
                "Invalid mathematical expression."
            )

        expression = expression.replace(
            "%",
            "/100"
        )

        result = eval(
            expression,
            {
                "__builtins__": None
            },
            {}
        )

        return str(result)

    except Exception:

        return (
            "Could not calculate "
            "the expression."
        )


# =========================================================
# MEMORY TOOLS
# =========================================================

def save_memory_tool(memory):

    return save_memory(memory)


def search_memory_tool(query):

    results = search_memory(
        query,
        limit=10
    )

    if not results:

        return "No matching memories found."

    output = []

    for row in results:

        memory_id, memory, created_at = row

        output.append(
            f"Memory {memory_id}: "
            f"{memory} "
            f"(saved {created_at})"
        )

    return "\n".join(output)


# =========================================================
# TOOL DEFINITIONS
# =========================================================

TOOLS = [

    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "rag_search",

            "description": (
                "Search the uploaded document "
                "for relevant information."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {

                        "type": "string",

                        "description": (
                            "A focused search query "
                            "for the document."
                        )
                    }
                },

                "required": [
                    "query"
                ]
            }
        }
    },

    # -----------------------------------------------------
    # CALCULATOR
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "calculator",

            "description": (
                "Perform mathematical calculations."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "expression": {

                        "type": "string",

                        "description": (
                            "Mathematical expression."
                        )
                    }
                },

                "required": [
                    "expression"
                ]
            }
        }
    },

    # -----------------------------------------------------
    # SAVE MEMORY
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "save_memory",

            "description": (
                "Save an important user-provided "
                "fact or preference into persistent "
                "memory. Only save information that "
                "is useful for future conversations."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "memory": {

                        "type": "string",

                        "description": (
                            "The useful fact to remember."
                        )
                    }
                },

                "required": [
                    "memory"
                ]
            }
        }
    },

    # -----------------------------------------------------
    # SEARCH MEMORY
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "search_memory",

            "description": (
                "Search persistent memory for "
                "information about the user or "
                "previously saved facts."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {

                        "type": "string",

                        "description": (
                            "What information to "
                            "look for in memory."
                        )
                    }
                },

                "required": [
                    "query"
                ]
            }
        }
    }
]


# =========================================================
# EXECUTE TOOL
# =========================================================

def execute_tool(
    tool_name,
    arguments,
    top_k
):

    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    if tool_name == "rag_search":

        query = arguments.get(
            "query",
            ""
        )

        results = rag_search(
            query,
            top_k
        )

        if not results:

            return (
                "No relevant information "
                "was found."
            )

        output = []

        for i, result in enumerate(
            results,
            start=1
        ):

            output.append(
                f"Result {i} "
                f"(similarity: "
                f"{result['score']:.3f})\n"
                f"{result['chunk']}"
            )

        return "\n\n".join(
            output
        )

    # -----------------------------------------------------
    # CALCULATOR
    # -----------------------------------------------------

    if tool_name == "calculator":

        expression = arguments.get(
            "expression",
            ""
        )

        return calculator(
            expression
        )

    # -----------------------------------------------------
    # SAVE MEMORY
    # -----------------------------------------------------

    if tool_name == "save_memory":

        memory = arguments.get(
            "memory",
            ""
        )

        return save_memory_tool(
            memory
        )

    # -----------------------------------------------------
    # SEARCH MEMORY
    # -----------------------------------------------------

    if tool_name == "search_memory":

        query = arguments.get(
            "query",
            ""
        )

        return search_memory_tool(
            query
        )

    return "Unknown tool."


# =========================================================
# AGENT
# =========================================================

def run_agent(
    user_question,
    top_k=4,
    max_iterations=6
):

    logs = []

    # -----------------------------------------------------
    # SYSTEM PROMPT
    # -----------------------------------------------------

    system_message = """
You are an intelligent Agentic AI assistant.

You have four tools:

1. rag_search
   Search the uploaded document.

2. calculator
   Perform mathematical calculations.

3. save_memory
   Save useful, non-sensitive information that the
   user explicitly provides and that could help in
   future conversations.

4. search_memory
   Search previously saved memories.

IMPORTANT MEMORY RULES:

- Do not save every message.
- Save information only when it is useful for future
  conversations.
- If the user explicitly says "remember this", save it.
- If the user asks about something they previously told
  you, search memory.
- Never claim to remember something unless memory
  actually contains it.
- Do not invent memories.

AGENT RULES:

- Decide which tool is needed.
- You may use multiple tools.
- After every tool result, reconsider the task.
- Do not use tools unnecessarily.
- Continue until enough information is available.
- Then provide a clear final answer.
- Never invent information from documents or memory.
"""

    messages = [
        {
            "role": "system",
            "content": system_message
        }
    ]

    # -----------------------------------------------------
    # ADD RECENT CONVERSATION
    # -----------------------------------------------------

    for message in (
        st.session_state.messages[-8:]
    ):

        messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )

    # -----------------------------------------------------
    # USER QUESTION
    # -----------------------------------------------------

    messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # =====================================================
    # AGENT LOOP
    # =====================================================

    for iteration in range(
        max_iterations
    ):

        logs.append(
            {
                "type": "iteration",
                "number": iteration + 1
            }
        )

        # -------------------------------------------------
        # CALL MODEL
        # -------------------------------------------------

        response = (
            client.chat.completions.create(

                model=MODEL_NAME,

                messages=messages,

                tools=TOOLS,

                tool_choice="auto",

                temperature=0.2,

                max_tokens=1400
            )
        )

        assistant_message = (
            response.choices[0].message
        )

        # -------------------------------------------------
        # FINAL ANSWER
        # -------------------------------------------------

        if not assistant_message.tool_calls:

            answer = (
                assistant_message.content
                or "No answer was generated."
            )

            logs.append(
                {
                    "type": "complete"
                }
            )

            return (
                answer,
                logs
            )

        # -------------------------------------------------
        # STORE ASSISTANT TOOL CALL
        # -------------------------------------------------

        tool_calls_for_message = []

        for tool_call in (
            assistant_message.tool_calls
        ):

            tool_calls_for_message.append(
                {
                    "id": tool_call.id,

                    "type": "function",

                    "function": {

                        "name": (
                            tool_call.function.name
                        ),

                        "arguments": (
                            tool_call.function.arguments
                        )
                    }
                }
            )

        messages.append(
            {
                "role": "assistant",

                "content": (
                    assistant_message.content
                    or ""
                ),

                "tool_calls":
                    tool_calls_for_message
            }
        )

        # -------------------------------------------------
        # EXECUTE TOOL CALLS
        # -------------------------------------------------

        for tool_call in (
            assistant_message.tool_calls
        ):

            tool_name = (
                tool_call.function.name
            )

            raw_arguments = (
                tool_call.function.arguments
            )

            try:

                arguments = json.loads(
                    raw_arguments
                )

            except Exception:

                arguments = {}

            logs.append(
                {
                    "type": "tool_call",

                    "tool": tool_name,

                    "arguments": arguments
                }
            )

            # ---------------------------------------------
            # RUN TOOL
            # ---------------------------------------------

            result = execute_tool(
                tool_name,
                arguments,
                top_k
            )

            logs.append(
                {
                    "type": "tool_result",

                    "tool": tool_name,

                    "result": result
                }
            )

            # ---------------------------------------------
            # SEND RESULT TO AGENT
            # ---------------------------------------------

            messages.append(
                {
                    "role": "tool",

                    "tool_call_id": (
                        tool_call.id
                    ),

                    "content": result
                }
            )

    # =====================================================
    # MAXIMUM ITERATIONS
    # =====================================================

    logs.append(
        {
            "type": "max_iterations"
        }
    )

    final_response = (
        client.chat.completions.create(

            model=MODEL_NAME,

            messages=messages,

            temperature=0.2,

            max_tokens=1400
        )
    )

    answer = (
        final_response
        .choices[0]
        .message
        .content
    )

    return (
        answer
        or "The agent could not finish the task.",
        logs
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Agent Settings")

    top_k = st.slider(
        "RAG results",
        1,
        8,
        4
    )

    max_iterations = st.slider(
        "Maximum agent steps",
        1,
        10,
        6
    )

    st.divider()

    # -----------------------------------------------------
    # DOCUMENT
    # -----------------------------------------------------

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX or TXT",
        type=[
            "pdf",
            "docx",
            "txt"
        ]
    )

    if uploaded_file:

        st.write(
            f"**File:** "
            f"{uploaded_file.name}"
        )

        if st.button(
            "🔄 Process Document",
            use_container_width=True
        ):

            with st.spinner(
                "Processing document..."
            ):

                try:

                    filename = (
                        uploaded_file.name.lower()
                    )

                    if filename.endswith(
                        ".pdf"
                    ):

                        text = extract_pdf(
                            uploaded_file
                        )

                    elif filename.endswith(
                        ".docx"
                    ):

                        text = extract_docx(
                            uploaded_file
                        )

                    else:

                        text = extract_txt(
                            uploaded_file
                        )

                    text = clean_text(
                        text
                    )

                    if not text:

                        st.error(
                            "No readable text "
                            "was found."
                        )

                    else:

                        chunks = create_chunks(
                            text
                        )

                        index = (
                            build_vector_database(
                                chunks
                            )
                        )

                        st.session_state.chunks = (
                            chunks
                        )

                        st.session_state.index = (
                            index
                        )

                        st.session_state.document_name = (
                            uploaded_file.name
                        )

                        st.success(
                            "Document processed "
                            "successfully!"
                        )

                except Exception as e:

                    st.error(
                        f"Processing error: {e}"
                    )

    st.divider()

    # -----------------------------------------------------
    # MEMORY
    # -----------------------------------------------------

    st.header("🧠 Persistent Memory")

    memories = get_memories(
        limit=20
    )

    st.write(
        f"Stored memories: "
        f"**{len(memories)}**"
    )

    if memories:

        with st.expander(
            "View memories"
        ):

            for row in memories:

                memory_id, memory, created_at = row

                st.write(
                    f"**#{memory_id}** "
                    f"{memory}"
                )

                st.caption(
                    created_at
                )

    if st.button(
        "🗑️ Clear All Memories",
        use_container_width=True
    ):

        clear_all_memories()

        st.success(
            "All memories cleared."
        )

        st.rerun()

    st.divider()

    # -----------------------------------------------------
    # TOOLS
    # -----------------------------------------------------

    st.header("🛠️ Agent Tools")

    st.success("🔎 RAG Search")

    st.success("🧮 Calculator")

    st.success("💾 Save Memory")

    st.success("🔍 Search Memory")

    st.divider()

    if st.session_state.index:

        st.info(
            f"Document: "
            f"{st.session_state.document_name}\n\n"
            f"Chunks: "
            f"{len(st.session_state.chunks)}"
        )

    else:

        st.warning(
            "No document loaded."
        )


# =========================================================
# DASHBOARD
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "RAG",
        "Active"
    )

with col2:

    st.metric(
        "Tools",
        "4"
    )

with col3:

    st.metric(
        "Agent",
        "Active"
    )

with col4:

    st.metric(
        "Memory",
        len(get_memories())
    )


st.divider()


# =========================================================
# CHAT HISTORY
# =========================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("agent_logs")
        ):

            with st.expander(
                "🧠 View Agent Activity"
            ):

                for log in (
                    message["agent_logs"]
                ):

                    if log["type"] == "iteration":

                        st.write(
                            f"🔄 Agent step "
                            f"{log['number']}"
                        )

                    elif log["type"] == "tool_call":

                        st.write(
                            f"🛠️ Tool: "
                            f"{log['tool']}"
                        )

                        st.write(
                            "Arguments:"
                        )

                        st.json(
                            log["arguments"]
                        )

                    elif log["type"] == "tool_result":

                        st.write(
                            f"📋 Result from "
                            f"{log['tool']}"
                        )

                        st.code(
                            log["result"]
                        )

                    elif log["type"] == "complete":

                        st.success(
                            "✅ Agent completed."
                        )

                    elif log["type"] == "max_iterations":

                        st.warning(
                            "Maximum agent steps "
                            "were reached."
                        )


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Talk to your agent..."
)


if user_question:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_question
        )

    # -----------------------------------------------------
    # AGENT
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🧠 Agent is working..."
        ):

            try:

                answer, logs = run_agent(
                    user_question,
                    top_k,
                    max_iterations
                )

                st.markdown(
                    answer
                )

                # -----------------------------------------
                # AGENT LOG
                # -----------------------------------------

                with st.expander(
                    "🧠 View Agent Activity"
                ):

                    for log in logs:

                        if log["type"] == "iteration":

                            st.write(
                                f"🔄 Agent step "
                                f"{log['number']}"
                            )

                        elif log["type"] == "tool_call":

                            st.write(
                                f"🛠️ Tool: "
                                f"{log['tool']}"
                            )

                            st.json(
                                log["arguments"]
                            )

                        elif log["type"] == "tool_result":

                            st.write(
                                f"📋 "
                                f"{log['tool']} result"
                            )

                            st.code(
                                log["result"]
                            )

                        elif log["type"] == "complete":

                            st.success(
                                "✅ Agent completed."
                            )

                        elif log["type"] == "max_iterations":

                            st.warning(
                                "Maximum agent steps "
                                "reached."
                            )

                # -----------------------------------------
                # SAVE
                # -----------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "agent_logs": logs
                    }
                )

            except Exception as e:

                error_message = (
                    f"AI request failed: {e}"
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "agent_logs": []
                    }
                )
