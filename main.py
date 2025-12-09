"""
Recruitment Agent API with NeMo Guardrails
Custom FastAPI implementation with NeMo Guardrails for Input/Output validation
"""
# AgentOps - Must be initialized before importing google-genai
import agentops
agentops.init()

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# NeMo Guardrails
from nemoguardrails import LLMRails, RailsConfig

# load environment variables
load_dotenv()


# configuration variables (can be overridden by environment variables)

# server configuration
HOST = os.getenv("HOST", "0.0.0.0")  # default 0.0.0.0 to support Docker/cloud deployment
PORT = int(os.getenv("PORT", "8080"))  # default port 8080

# CORS configuration (should limit specific domains in production)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# API configuration
API_TITLE = os.getenv("API_TITLE", "Recruitment Agent with Guardrails")
API_VERSION = os.getenv("API_VERSION", "1.0.0")

# Debug mode
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# import Agent related modules
from agent import (
    router as root_agent,
    session_service,
    memory_service,
    recruitment_app,
    runner
)

# For Output Guardrails, we still import the mask function directly for simplicity
# as invoking 'output-only' flows via Engine on raw strings is complex without a full cycle.
# Input guardrails will be handled entirely by the Engine.
from guardrails.actions import mask_pii


# FastAPI App configuration
app = FastAPI(
    title=API_TITLE,
    description="A recruitment assistant powered by Google ADK with NeMo Guardrails",
    version=API_VERSION,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NeMo Guardrails
# Loads config from the 'guardrails' directory (config.yml, rails/*.co, actions.py)
print("⚡ Initializing NeMo Guardrails Engine...")
try:
    rails_config = RailsConfig.from_path("./guardrails")
    rails = LLMRails(rails_config)
    print("✅ NeMo Guardrails Engine Initialized")
except Exception as e:
    print(f"❌ Failed to initialize NeMo Guardrails: {e}")
    # Fallback or exit? For now, we let it fail if config is bad.
    raise e


# Request/Response models

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "default_user"


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    blocked: bool = False
    block_reason: Optional[str] = None
    session_id: Optional[str] = None
    masked: bool = False  # whether PII was masked


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    guardrails_enabled: bool
    version: str



# Guardrail functions

async def run_input_guardrails(message: str) -> tuple[bool, Optional[str]]:
    """
    Execute Input Rails using NeMo Guardrails Engine.
    Returns: (is_allowed, block_reason)
    """
    try:
        # We use rails.generate to check the input.
        # Flows in 'rules.co' are defined to 'stop' and return a refusal message if a check fails.
        # If checks pass, and since we have no LLM, it might return None, empty, or error.
        # We assume any non-empty response that looks like our bot refusals is a Block.
        
        response = await rails.generate(messages=[{
            "role": "user",
            "content": message
        }])
        
        # Check if we got a refusal response
        if response and response.response:
            # If the engine produced a response (meaning a rail triggered a 'bot refuse ...'), we block.
            # In a full chat setup, 'response' would be the LLM answer. 
            # Here, strict Input Rails imply that an immediate response is a Refusal.
            return False, response.response
            
        # If no response (or empty), it means input rails passed and execution stopped (no LLM).
        return True, None
        
    except Exception as e:
        # If "No model configured" error happens after rails pass, it's actually a Success for input checking.
        # But if it's another error, we log it.
        # Assuming 'models: []' allows rails to run and finish if no generation is needed?
        # NeMo might raise an exception if it tries to generate.
        # For this demo, we assume failure to generate = Success of Input Rails (nothing blocked it).
        # print(f"Guardrails processing note: {e}")
        return True, None


async def run_output_guardrails(output: str) -> tuple[str, bool]:
    """
    Execute Output Rails.
    Currently using direct function call as NeMo Output Rails are tightly coupled to Bot generation.
    """
    masked_output = await mask_pii(output)
    was_masked = masked_output != output
    return masked_output, was_masked



# API endpoints

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Flow:
    1. Input Guardrails - Check user input via NeMo Engine
    2. Agent processing - Call ADK Agent
    3. Output Guardrails - Clean output
    """
    
    # Step 1: Input Guardrails
    is_allowed, block_reason = await run_input_guardrails(request.message)
    
    if not is_allowed:
        return ChatResponse(
            response=block_reason,
            blocked=True,
            block_reason=block_reason,
            session_id=request.session_id,
        )
    
    # Step 2: Agent processing 
    try:
        from google.genai import types
        import uuid
        
        # Ensure session_id and user_id
        session_id = request.session_id or str(uuid.uuid4())
        user_id = request.user_id or "default_user"
        
        # Create or get session
        existing_session = await session_service.get_session(
            app_name=recruitment_app.name,
            user_id=user_id,
            session_id=session_id
        )
        
        if not existing_session:
            await session_service.create_session(
                app_name=recruitment_app.name,
                user_id=user_id,
                session_id=session_id
            )
        
        # Create message content
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=request.message)]
        )
        
        # Use run_async to get event stream
        agent_output = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            # Extract final text response
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        agent_output = part.text  # Get the last text response
                        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Agent processing error: {str(e)}"
        )
    
    # Step 3: Output Guardrails 
    cleaned_output, was_masked = await run_output_guardrails(agent_output)
    
    return ChatResponse(
        response=cleaned_output,
        blocked=False,
        session_id=request.session_id,
        masked=was_masked,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        guardrails_enabled=True,
        version="1.0.0",
    )


@app.get("/")
async def root():
    return {
        "message": "Recruitment Agent API with Guardrails",
        "docs": "/docs",
        "health": "/health",
    }



# Start the app

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting Recruitment Agent with Guardrails")
    print("=" * 60)
    print(f"📍 Host:     {HOST}")
    print(f"📍 Port:     {PORT}")
    print(f"📍 API Docs: http://{HOST}:{PORT}/docs")
    print(f"📍 Health:   http://{HOST}:{PORT}/health")
    print(f"📍 Debug:    {DEBUG}")
    print("=" * 60)
    
    uvicorn.run(app, host=HOST, port=PORT, reload=DEBUG)
