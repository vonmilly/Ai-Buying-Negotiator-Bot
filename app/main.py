import os
import subprocess
import time
import csv
from io import StringIO
from typing import List, Dict, Any
import re # Import regex module

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# --- Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Path to llama.cpp executable and model within the Docker container
LLAMA_CPP_EXECUTABLE = "./llama.cpp/main"
MISTRAL_MODEL_PATH = "./llama.cpp/models/mistral.gguf" # Ensure your model is copied here during Docker build

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Self-Hosted AI Negotiator",
    description="API for a text-based sales negotiation AI powered by Mistral 7B and Qdrant.",
    version="1.0.0"
)

# --- Initialize Models and Clients ---
model: SentenceTransformer = None
qdrant: QdrantClient = None

# Qdrant Collection Names
CONVERSATION_COLLECTION = "sales_conversations"
PRODUCT_COLLECTION = "product_knowledge" # This collection will now store graded phone data

# Define the expected grade headers from your CSV for graded phones
GRADE_HEADERS = ["Grade A", "Grade B+", "Grade B", "Grade C", "Grade D", "DOA"]

@app.on_event("startup")
async def startup_event():
    """
    Initializes SentenceTransformer and Qdrant client on application startup.
    Ensures Qdrant collections are ready.
    """
    global model, qdrant
    print(f"Initializing SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2') # This model outputs 384-dimensional vectors
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    try:
        # Ensure CONVERSATION_COLLECTION exists
        if not qdrant.collection_exists(collection_name=CONVERSATION_COLLECTION):
            print(f"Creating '{CONVERSATION_COLLECTION}' collection in Qdrant...")
            qdrant.recreate_collection(
                collection_name=CONVERSATION_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        else:
            print(f"'{CONVERSATION_COLLECTION}' collection already exists.")

        # Ensure PRODUCT_COLLECTION exists
        if not qdrant.collection_exists(collection_name=PRODUCT_COLLECTION):
            print(f"Creating '{PRODUCT_COLLECTION}' collection in Qdrant...")
            qdrant.recreate_collection(
                collection_name=PRODUCT_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        else:
            print(f"'{PRODUCT_COLLECTION}' collection already exists.")

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
    Encodes a message and stores it in the Qdrant vector database (conversation history).
    """
    if model is None or qdrant is None:
        print("Error: Model or Qdrant client not initialized.")
        return

    try:
        vector = model.encode(content).tolist()
        qdrant.upsert(
            collection_name=CONVERSATION_COLLECTION,
            points=[PointStruct(id=int(time.time()*1000), vector=vector, payload={"session": session_id, "role": role, "text": content})]
        )
        print(f"Stored conversation message for session {session_id}, role {role}.")
    except Exception as e:
        print(f"Error storing conversation message in Qdrant: {e}")

def retrieve_conversation_context(session_id: str, query_text: str, limit: int = 5) -> str:
    """
    Retrieves relevant past messages from Qdrant for a given session.
    """
    if model is None or qdrant is None:
        print("Error: Model or Qdrant client not initialized.")
        return ""

    try:
        query_vector = model.encode(query_text).tolist()
        search_result = qdrant.query(
            collection_name=CONVERSATION_COLLECTION,
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
        context = "\n".join([hit.payload["text"] for hit in search_result.points])
        print(f"Retrieved conversation context for session {session_id}: {len(search_result.points)} messages.")
        return context
    except Exception as e:
        print(f"Error retrieving conversation context from Qdrant: {e}")
        return ""

def retrieve_product_info(query_text: str, limit: int = 5) -> str:
    """
    Retrieves relevant product information from the product knowledge Qdrant collection.
    This function is now adapted to handle graded phone data and general product data.
    """
    if model is None or qdrant is None:
        print("Error: Model or Qdrant client not initialized.")
        return ""

    try:
        query_vector = model.encode(query_text).tolist()
        search_result = qdrant.query(
            collection_name=PRODUCT_COLLECTION,
            query_vector=query_vector,
            limit=limit
        )
        product_info_list = []
        for hit in search_result.points:
            payload = hit.payload
            # Construct a more detailed and flexible description from the payload
            # Check for specific keys to determine if it's a graded phone, general product, or a guideline
            if 'type' in payload and payload['type'] == 'graded_phone_price':
                # This is a graded phone price entry
                product_info_list.append(
                    f"Product: {payload.get('full_variant_name')}, " # Use full_variant_name for clarity
                    f"Price: ${payload.get('price')}"
                )
            elif 'type' in payload and payload['type'] == 'grading_guideline':
                # This is a grading guideline entry
                product_info_list.append(
                    f"Grading Guideline: {payload.get('name')}: {payload.get('description')}"
                )
            elif 'type' in payload and payload['type'] == 'general_product':
                # Fallback for other product types, e.g., the 'new_iphones.csv' format
                product_info_list.append(
                    f"Product: {payload.get('brand')} {payload.get('model_name')}, "
                    f"Price: ${payload.get('price')}, "
                    f"Storage: {payload.get('storage_gb')}GB, "
                    f"Color: {payload.get('color')}, "
                    f"Features: {payload.get('features')}"
                )


        context = "\n".join(product_info_list)
        print(f"Retrieved product info: {len(search_result.points)} items.")
        return context
    except Exception as e:
        print(f"Error retrieving product info from Qdrant: {e}")
        return ""

# --- FastAPI Endpoint for Product Ingestion (Now handles graded data and guidelines) ---
@app.post("/ingest-data") # Renamed endpoint for broader ingestion
async def ingest_data(file: UploadFile = File(...)):
    """
    Ingests data from a CSV file into the product_knowledge Qdrant collection.
    This endpoint is now flexible to handle:
    1. Graded phone data (e.g., iPhone Model, Grade A, Grade B+, etc.)
    2. General product data (e.g., model_name,brand,price,storage_gb,color,features)
    3. Grading guidelines (e.g., Guideline Name, Description)
    """
    if qdrant is None or model is None:
        raise HTTPException(status_code=503, detail="Service not initialized. Please try again later.")

    try:
        content = await file.read()
        csv_string = StringIO(content.decode('utf-8'))
        
        # Read the first line to determine the CSV type
        first_line = csv_string.readline().strip()
        csv_string.seek(0) # Reset stream position to the beginning

        # Determine CSV type based on headers
        # Prioritize graded phone headers as they are very specific
        # Check for "iPhone Model" and any of the GRADE_HEADERS
        if "iPhone Model" in first_line and any(grade_header in first_line for grade_header in GRADE_HEADERS):
            # This is a graded phone data CSV
            return await _ingest_graded_phones(csv_string)
        elif "Guideline Name" in first_line and "Description" in first_line:
            # This is a grading guidelines CSV
            return await _ingest_grading_guidelines(csv_string)
        elif "model_name" in first_line and "brand" in first_line and "price" in first_line:
            # This is the general product data CSV (like new_iphones.csv)
            return await _ingest_general_products(csv_string)
        else:
            raise HTTPException(status_code=400, detail="Unrecognized CSV format. Please ensure headers match expected types.")

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error ingesting data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest data: {str(e)}")

async def _ingest_graded_phones(csv_string: StringIO):
    """Helper to ingest CSVs with graded phone prices."""
    reader = csv.DictReader(csv_string)
    points_to_upsert = []
    current_id_base = int(time.time() * 1000000) # Use a larger multiplier for more unique IDs

    for row_idx, row in enumerate(reader):
        # Get model_info directly from "iPhone Model" column
        model_info = row.get("iPhone Model", "").strip()

        if not model_info:
            print(f"Skipping row {row_idx + 1} due to missing model information in 'iPhone Model' column.")
            continue

        # Extract storage and condition from model_info
        # This regex now handles both "256GB" and single digits like "2" (assuming 256GB)
        storage_gb = 0
        storage_match = re.search(r'(\d+GB|\d+)', model_info, re.IGNORECASE)
        if storage_match:
            storage_str = storage_match.group(1).replace('GB', '')
            if storage_str == '2': # Common shorthand for 256GB
                storage_gb = 256
            elif storage_str == '5': # Common shorthand for 512GB
                storage_gb = 512
            elif storage_str == '1': # Common shorthand for 1TB (1000GB)
                storage_gb = 1000
            else:
                try:
                    storage_gb = int(storage_str)
                except ValueError:
                    pass # Keep as 0 if conversion fails

        condition_part = ""
        if "Unlocked" in model_info:
            condition_part = "Unlocked"
        elif "Carrier Locked" in model_info:
            condition_part = "Carrier Locked"
        elif "Carrier" in model_info: # Fallback for partial "Carrier"
            condition_part = "Carrier Locked"
        elif "CDMA" in model_info:
            condition_part = "Carrier Locked"
        
        # Clean up base model name (e.g., "iPhone 16 Pro Max")
        # Remove storage and condition parts to get just the model name
        base_model_name = model_info.replace(f"{storage_gb}GB", "").replace("Unlocked", "").replace("Carrier Locked", "").replace("Carrier", "").strip()
        base_model_name = re.sub(r'Unlo\w*$', '', base_model_name, flags=re.IGNORECASE).strip() # Remove partial "Unlo"
        base_model_name = re.sub(r'Carri\w*$', '', base_model_name, flags=re.IGNORECASE).strip() # Remove partial "Carri"
        base_model_name = base_model_name.replace("  ", " ").strip() # Clean up extra spaces
        
        # Ensure 'iPhone' is present if it's an iPhone model
        if "iPhone" not in base_model_name and "iPhone" in model_info:
            base_model_name = f"iPhone {base_model_name}".strip()


        for i, grade_header in enumerate(GRADE_HEADERS):
            price_str = row.get(grade_header, "").replace('$', '').strip()
            if not price_str:
                print(f"Skipping {grade_header} for {model_info} due to missing price.")
                continue
            try:
                price = float(price_str)
            except ValueError:
                print(f"Skipping {grade_header} for {model_info} due to invalid price: '{price_str}'.")
                continue

            # Create a unique model_name for each graded entry
            full_variant_name = f"{base_model_name} {storage_gb}GB ({condition_part}, Grade {grade_header})"
            
            description = (
                f"A {condition_part} {base_model_name} with {storage_gb}GB storage, "
                f"graded as {grade_header}, is priced at ${price}."
            )
            
            vector = model.encode(description).tolist()
            
            payload = {
                "model_name": base_model_name, # Keep base model name for easier grouping
                "full_variant_name": full_variant_name, # Store the unique descriptive name
                "brand": "Apple", # Assuming Apple for these iPhones
                "price": price,
                "storage_gb": storage_gb,
                "condition": condition_part,
                "grade": grade_header,
                "description": description, # Store the full description for retrieval context
                "type": "graded_phone_price" # Explicitly mark this type of data
            }
            points_to_upsert.append(
                PointStruct(
                    id=current_id_base + (row_idx * len(GRADE_HEADERS)) + i, # Unique ID for each point
                    vector=vector,
                    payload=payload
                )
            )
    
    if points_to_upsert:
        qdrant.upsert(
            collection_name=PRODUCT_COLLECTION,
            points=points_to_upsert,
            wait=True
        )
        print(f"Successfully ingested {len(points_to_upsert)} graded phone records into {PRODUCT_COLLECTION}.")
        return JSONResponse(content={"message": f"Successfully ingested {len(points_to_upsert)} graded phone records."}, status_code=200)
    else:
        return JSONResponse(content={"message": "No valid graded phone records found in CSV."}, status_code=400)

async def _ingest_general_products(csv_string: StringIO):
    """Helper to ingest CSVs with general product data (like new_iphones.csv)."""
    reader = csv.DictReader(csv_string)
    points_to_upsert = []
    current_id_base = int(time.time() * 1000000) + 1000000 # Offset IDs to avoid collision with graded phones

    for i, row in enumerate(reader):
        # Basic validation and type conversion for general products
        try:
            row['price'] = float(row['price'])
            row['storage_gb'] = int(row['storage_gb'])
        except (ValueError, KeyError) as ve:
            print(f"Skipping row {i+1} due to invalid data type or missing key: {ve} in row {row}")
            continue

        # Create a descriptive text for embedding
        description = (
            f"{row.get('brand', 'Unknown')} {row.get('model_name', 'Unknown Model')} costs ${row.get('price', 0)}. "
            f"It has {row.get('storage_gb', 0)}GB storage and comes in {row.get('color', 'various colors')}. "
            f"Key features include: {row.get('features', 'No specific features listed')}."
        )
        vector = model.encode(description).tolist()
        
        # Ensure all expected keys are present in payload, even if empty in CSV
        payload = {
            "model_name": row.get('model_name', ''),
            "brand": row.get('brand', ''),
            "price": row.get('price', 0.0),
            "storage_gb": row.get('storage_gb', 0),
            "color": row.get('color', ''),
            "features": row.get('features', ''),
            "type": "general_product" # Explicitly mark this type of data
        }

        points_to_upsert.append(
            PointStruct(
                id=current_id_base + i,
                vector=vector,
                payload=payload
            )
        )
    
    if points_to_upsert:
        qdrant.upsert(
            collection_name=PRODUCT_COLLECTION,
            points=points_to_upsert,
            wait=True
        )
        print(f"Successfully ingested {len(points_to_upsert)} general product records into {PRODUCT_COLLECTION}.")
        return JSONResponse(content={"message": f"Successfully ingested {len(points_to_upsert)} general product records."}, status_code=200)
    else:
        return JSONResponse(content={"message": "No valid general product records found in CSV."}, status_code=400)


async def _ingest_grading_guidelines(csv_string: StringIO):
    """Helper to ingest CSVs with grading guidelines."""
    reader = csv.DictReader(csv_string)
    points_to_upsert = []
    current_id_base = int(time.time() * 1000000) + 2000000 # Offset IDs to avoid collision

    for i, row in enumerate(reader):
        name = row.get("Guideline Name", "").strip()
        description = row.get("Description", "").strip()

        if not name or not description:
            print(f"Skipping guideline row {i+1} due to missing name or description.")
            continue

        # Use the description itself for embedding
        vector = model.encode(description).tolist()
        
        payload = {
            "type": "grading_guideline", # Custom type to identify this data
            "name": name,
            "description": description
        }
        points_to_upsert.append(
            PointStruct(
                id=current_id_base + i,
                vector=vector,
                payload=payload
            )
        )
    
    if points_to_upsert:
        qdrant.upsert(
            collection_name=PRODUCT_COLLECTION,
            points=points_to_upsert,
            wait=True
        )
        print(f"Successfully ingested {len(points_to_upsert)} grading guideline records into {PRODUCT_COLLECTION}.")
        return JSONResponse(content={"message": f"Successfully ingested {len(points_to_upsert)} grading guideline records."}, status_code=200)
    else:
        return JSONResponse(content={"message": "No valid grading guideline records found in CSV."}, status_code=400)


# --- FastAPI Endpoint for Negotiation ---
@app.post("/negotiate")
async def negotiate(request: Request):
    """
    Handles sales negotiation requests.
    - Retrieves product context from Qdrant.
    - Retrieves conversation context from Qdrant.
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

        # 1. Retrieve relevant product information (now handles graded data and guidelines)
        product_context = retrieve_product_info(user_input)
        if product_context:
            print(f"Found product context:\n{product_context}")
        else:
            print("No specific product context found for this query.")

        # 2. Retrieve past conversation context
        conversation_context = retrieve_conversation_context(session_id, user_input)
        if conversation_context:
            print(f"Found conversation context:\n{conversation_context}")
        else:
            print("No past conversation context for this session.")


        # Format prompt for the LLM
        # You can refine this prompt heavily for better negotiation behavior
        # Provide product context first, then conversation history, then user query
        final_prompt = (
            "You are a helpful sales assistant. Your goal is to negotiate a sale successfully. "
            "Be polite, persuasive, and aim for a win-win outcome. Always respond as the AI.\n\n"
        )
        if product_context:
            final_prompt += f"Available Product Information:\n{product_context}\n\n"
        if conversation_context:
            final_prompt += f"Past Conversation History:\n{conversation_context}\n\n"

        final_prompt += f"User: {user_input}\nAI:"
        print(f"Formatted prompt for LLM:\n{final_prompt[:1000]}...") # Print first 1000 chars

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

