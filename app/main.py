from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import subprocess
import os
import time

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Path to llama.cpp executable and model within the Docker container
LLAMA_CPP_EXECUTABLE = "./llama.cpp/main"
MISTRAL_MODEL_PATH = "./llama.cpp/models/mistral-7b-instruct-v0.2.Q2_K.gguf" # Ensure your model is copied here during Docker build

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Self-Hosted AI Negotiator",
    description="API for a text-based sales negotiation AI powered by Mistral 7B and Qdrant.",
    version="1.0.0"
)

# --- Initialize Models and Clients ---
model: SentenceTransformer = None
qdrant: QdrantClient = None

@app.on_event("startup")
async def startup_event():
    """
    Initializes SentenceTransformer and Qdrant client on application startup.
    Ensures Qdrant collection is ready.
    """
    global model, qdrant
    print(f"Initializing SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    try:
        # Check if collection exists, if not, create it
        if not qdrant.collection_exists(collection_name="sales_memory"):
            print("Creating 'sales_memory' collection in Qdrant...")
            qdrant.recreate_collection(
                collection_name="sales_memory",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        else:
            print("'sales_memory' collection already exists.")
    except Exception as e:
        print(f"Error connecting to Qdrant or creating collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to Qdrant: {e}")

    # Basic check for llama.cpp executable and model existence
    if not os.path.exists(LLAMA_CPP_EXECUTABLE):
        print(f"Error: llama.cpp executable not found at {LLAMA_CPP_EXECUTABLE}")
        raise HTTPException(status_code=500, detail=f"llama.cpp executable not found at {LLAMA_CPP_EXECUTABLE}")
    if not os.path.exists(MISTRAL_MODEL_PATH):
        print(f"Error: Mistral model not found at {MISTRAL_MODEL_PATH}")
        raise HTTPException(status_code=500, detail=f"Mistral model not found at {MISTRAL_MODEL_PATH}")

    print("Application startup complete.")

# --- Helper Functions ---
def store_message(session_id: str, role: str, content: str):
    """
    Encodes a message and stores it in the Qdrant vector database.
    """
    if model is None or qdrant is None:
        print("Error: Model or Qdrant client not initialized.")
        return

    try:
        vector = model.encode(content).tolist()
        qdrant.upsert(
            collection_name="sales_memory",
            points=[PointStruct(id=int(time.time()*1000), vector=vector, payload={"session": session_id, "role": role, "text": content})]
        )
        print(f"Stored message for session {session_id}, role {role}.")
    except Exception as e:
        print(f"Error storing message in Qdrant: {e}")


def retrieve_context(session_id: str, query_text: str, limit: int = 5) -> str:
    """
    Retrieves relevant past messages from Qdrant for a given session.
    """
    if model is None or qdrant is None:
        print("Error: Model or Qdrant client not initialized.")
        return ""

    try:
        query_vector = model.encode(query_text).tolist()
        search_result = qdrant.search(
            collection_name="sales_memory",
            query_vector=query_vector,
            limit=limit,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="session",
                        match=MatchValue(value=session_id)
                    )
                ]
            )
        )
        context = "\n".join([hit.payload["text"] for hit in search_result])
        print(f"Retrieved context for session {session_id}: {len(search_result)} messages.")
        return context
    except Exception as e:
        print(f"Error retrieving context from Qdrant: {e}")
        return ""

# --- FastAPI Endpoint for Negotiation ---
@app.post("/negotiate")
async def negotiate(request: Request):
    """
    Handles sales negotiation requests.
    - Retrieves context from Qdrant.
    - Formats prompt for the LLM.
    - Runs llama.cpp for inference.
    - Stores user and AI messages in Qdrant.
    """
    try:
        data = await request.json()
        session_id = data.get("session_id")
        user_input = data.get("prompt")

        if not session_id or not user_input:
            raise HTTPException(
                status_code=400,
                detail="session_id and prompt are required"
            )

        print(f"Received negotiation request for session {session_id}: '{user_input}'")

        # Retrieve context from memory
        past_context = retrieve_context(session_id, user_input)

        # Format prompt for the LLM
        # You can refine this prompt heavily for better negotiation behavior
        final_prompt = f"You are a helpful sales assistant. Your goal is to negotiate a sale successfully. Be polite, persuasive, and aim for a win-win outcome. Always respond as the AI.\n\nContext:\n{past_context}\n\nUser: {user_input}\nAI:"
        print(f"Formatted prompt for LLM:\n{final_prompt[:500]}...") # Print first 500 chars

        # Run llama.cpp for inference
        try:
            command = [
                LLAMA_CPP_EXECUTABLE,
                "-m", MISTRAL_MODEL_PATH,
                "-p", final_prompt,
                "-n", "512", # Max tokens to generate
                "--temp", "0.7", # Sampling temperature
                "--top-p", "0.9", # Top-p sampling
                "--batch-size", "512", # Batch size for processing
                "--ctx-size", "2048", # Context window size
                "--no-display-prompt" # Do not display the prompt in stdout
            ]
            print(f"Executing llama.cpp command: {' '.join(command)}")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True, # Raise CalledProcessError for non-zero exit codes
                cwd="/app/llama.cpp" # Run llama.cpp from its directory
            )
            response_text = result.stdout.strip()
            print(f"LLM Response: {response_text}")

        except subprocess.CalledProcessError as e:
            print(f"LLM inference failed: {e.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"LLM inference failed: {e.stderr}"
            )
        except FileNotFoundError:
            print(f"llama.cpp executable not found. Check path: {LLAMA_CPP_EXECUTABLE}")
            raise HTTPException(
                status_code=500,
                detail="llama.cpp executable not found inside container."
            )
        except Exception as e:
            print(f"An unexpected error occurred during LLM inference: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected error occurred during LLM inference: {str(e)}"
            )

        # Store user and AI messages in vector DB for future context
        store_message(session_id, "user", user_input)
        store_message(session_id, "assistant", response_text)

        return JSONResponse(content={"response": response_text}, status_code=200)

    except HTTPException as e:
        raise e # Re-raise HTTPExceptions
    except Exception as e:
        print(f"An unhandled error occurred in the /negotiate endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error in negotiation endpoint: {str(e)}"
        )

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running and connected to Qdrant.
    """
    try:
        if qdrant:
            qdrant_health = qdrant.get_cluster_info()
            return JSONResponse(
                content={"status": "healthy", "qdrant_status": qdrant_health},
                status_code=200
            )
        else:
            return JSONResponse(
                content={"status": "initializing", "message": "Qdrant client not yet initialized."},
                status_code=503
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unhealthy: Failed to connect to Qdrant or get info: {e}"
        )

