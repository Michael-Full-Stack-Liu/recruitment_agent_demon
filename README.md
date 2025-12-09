# Containerized Recruitment Agent

A demon recruitment AI agent backed by **Google ADK** and protected by **NeMo Guardrails**. Fully containerized for easy deployment.

## � Features

- **AI Core**: Google ADK (Gemini Models).
- **Safety**: NeMo Guardrails for input/output validation.
- **Observability**: Full tracing with **AgentOps**.
- **API**: High-performance **FastAPI** backend.
- **Deploy**: Docker & Docker Compose ready.

## 🛠️ Setup & Run

### 1. Environment Configuration
Create a `.env` file in the root directory with your API keys:

```env
GOOGLE_API_KEY=your_gemini_api_key
AGENTOPS_API_KEY=your_agentops_api_key

# Keep HOST=0.0.0.0 for Docker containers (allows external access)
HOST=0.0.0.0
PORT=8080
```

### 2. Start Services
Run the entire stack with a single command:

```bash
docker compose up --build -d
```

### 3. Verify Deployment
- **API Health**: [http://localhost:8080/health](http://localhost:8080/health)
- **Interactive Docs**: [http://localhost:8080/docs](http://localhost:8080/docs)

## 📡 Usage Example

**Send a Chat Request:**
```bash
curl -X POST "http://localhost:8080/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "I want to hire a Senior Python Engineer"}'
```

## 🔍 Observability & Logs
To view real-time logs and access AgentOps session traces:

```bash
docker logs -f recruitment-agent
```


