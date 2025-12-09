import asyncio
import os
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService,InMemorySessionService
from google.adk.tools import google_search
from google.adk.apps.app import App, ResumabilityConfig,EventsCompactionConfig,ContextCacheConfig
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import FunctionTool,AgentTool
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.tools import McpToolset

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemma-3-27b"

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

session_service = InMemorySessionService()
memory_service= InMemoryMemoryService()

context_cache_config=ContextCacheConfig(
        min_tokens=1024,      # min token, reduce waste, caching has it's own cost
        ttl_seconds=3600,     # valid for 1 hour,short time tasks.86400 for long time tasks.
        cache_intervals=100   # max use 100 times, in case cache out of date.
    )

events_compact_config= EventsCompactionConfig(
         # the number of events to compact, the more complex the app, 
         # the more events ouured in one question from user, normally one agent one event.
        compaction_interval=70,  
        overlap_size=7,
    )

async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

def human_approve_jd(approve: bool):
    """
    Approve or reject the candidate.
    Args:
        approve: True to approve, False to reject
    """
    if approve:
        return {
            "status": "approved",
            "message": "Job details approved.",
        }
    else:
        return {
            "status": "rejected",
            "message": "Job details rejected.",
        }

# 1, job detail creater
jd_creater=Agent(
    name="jd_creater",
    model=Gemini(model=MODEL_NAME,retry_options=retry_config),
    instruction="""You are a job description creator. Based on user input and conversation history, identify job requirements.
                RULES:
                1. If there is no existing job description, create one with: job title, responsibilities, requirements, qualifications, and salary range.
                2. After creating the JD, use the [human_approve_jd] tool to ask for user approval.
                3. If user approves, confirm the JD is saved and suggest next steps (e.g., "You can now provide resumes for screening").
                4. If user rejects, ask for specific feedback and revise accordingly.
                5. If a JD already exists, do not create a new one unless user explicitly requests to update it.
                """,
    output_key="job_details",
    tools=[FunctionTool(human_approve_jd,require_confirmation=True)],
    after_agent_callback=auto_save_to_memory,
)

# 2. Resume Screener
screener = Agent(
    name="ResumeScreener",
    model=Gemini(model=MODEL_NAME,retry_options=retry_config),
    instruction="""You are an expert HR assistant. Analyze the resume provided in the input text.
                Extract and return ONLY valid JSON in the following schema:
                {
                    "name": "",
                    "phone": "",
                    "email": "",
                    "location": "",
                    "years_of_experience": 0,
                    "education": [...],
                    "work_experience": [...],
                    "skills": [],
                    "summary": ""
                }
                RULES:
                - If any field cannot be extracted, set it to null (for strings/arrays) or 0 (for numbers).
                - Never add commentary or explanations outside the JSON.
                - Ensure JSON is properly formatted and parseable.
                """,
    output_key="structured_resume",
    after_agent_callback=auto_save_to_memory,
)

# 3. Candidate Matcher (parallel)
matcher = Agent(
    name="CandidateMatcher",
    model=Gemini(model=MODEL_NAME,retry_options=retry_config),
    instruction="""You are a senior headhunter. Compare the structured resume against the job details provided in the context.
                EVALUATION CRITERIA:
                - Skills match (40%): How well candidate's skills align with requirements
                - Experience match (30%): Years and relevance of work experience
                - Education match (20%): Degree level and field relevance
                - Location/Availability (10%): Geographic and timing considerations
                You MUST return a valid JSON object:
                {
                    "match_score": <0-100>,
                    "reason": "<brief explanation of score>",
                    "recommend_proceed": <true/false>
                }
                If information is missing, set fields to null or 0. Never add commentary outside JSON.
                """,
    output_key="matching_result",
)

# 4. Integrity Checker (parallel)
integrity_checker = Agent(
    name="IntegrityChecker",
    model=Gemini(model=MODEL_NAME,retry_options=retry_config),
    instruction="""You are a resume integrity and fraud detection specialist. Analyze the structured resume for potential red flags.
                CHECK FOR:
                1. Age vs Experience mismatch (e.g., 22 years old claiming 10 years of experience)
                2. Overlapping employment dates
                3. Exaggerated titles or responsibilities
                4. Education timeline inconsistencies
                5. Gaps in employment without explanation
                You MUST return a valid JSON object:
                {
                    "integrity_score": <0-100, higher = more trustworthy>,
                    "risk_level": "<low/medium/high>",
                    "flags": ["<list of specific concerns found>"],
                    "recommendation": "<proceed/review/block>"
                }
                SCORING GUIDE:
                - 80-100: Low risk, proceed
                - 50-79: Medium risk, needs human review
                - 0-49: High risk, recommend blocking
                If information is missing, set fields to null or 0.
                """,
    output_key="integrity_report",
)

# 5. Bias Checker (parallel)
bias_checker = Agent(
    name="BiasChecker",
    model=Gemini(model=MODEL_NAME,retry_options=retry_config),
    instruction="""You are a DEI (Diversity, Equity & Inclusion) compliance officer. Review the evaluation process for potential bias in candidate assessment.
                CHECK FOR BIAS IN:
                1. Age discrimination (penalizing candidates over/under certain ages)
                2. Gender bias in language or scoring
                3. Racial or ethnic stereotyping
                4. Educational institution prestige bias
                5. Geographic or nationality bias
                You MUST return a valid JSON object:
                {
                    "bias_score": <0.0-1.0, higher = more bias detected>,
                    "detected_biases": ["<list of specific biases found>"],
                    "requires_human_review": <true/false>
                }
                THRESHOLDS:
                - bias_score < 0.2: Acceptable, no review needed
                - bias_score 0.2-0.5: Marginal, recommend review
                - bias_score > 0.5: Significant bias, requires human review
                If no bias detected, return bias_score: 0.0 and empty detected_biases array.
                """,
    output_key="bias_report",
)

resume_checker = SequentialAgent(
    name="ResumeChecker",
    sub_agents=[screener,matcher,integrity_checker,bias_checker]
)

def human_approve(approve: bool):
    """
    Approve or reject the candidate.
    Args:
        approve: True to approve, False to reject
    """
    if approve:
        return {
            "status": "approved",
            "message": "Candidate approved for interview.",
        }
    else:
        return {
            "status": "rejected",
            "message": "Candidate rejected.",
        }
    
# 6. Final Scheduler
scheduler = Agent(
    name="FinalScheduler",
    model=Gemini(model=MODEL_NAME,retry_options=retry_config),
    instruction="""You are a senior recruitment coordinator. Your role is to aggregate and synthesize the following reports from context:
                1. matching_result - Candidate-job fit assessment
                2. integrity_report - Resume authenticity check
                3. bias_report - DEI compliance review
                DECISION LOGIC:
                - If integrity_report risk_level is "high" OR bias_report requires_human_review is true → Recommend REJECT or REVIEW
                - If match_score < 50 → Recommend REJECT
                - If match_score >= 70 AND integrity is acceptable → Recommend APPROVE
                ACTIONS:
                1. Summarize key findings from all three reports.
                2. Provide a clear recommendation with reasoning.
                3. Use the [human_approve] tool to let the human make the final decision.
                4. If approved: Suggest next steps (e.g., schedule interview, prepare offer).
                5. If rejected: Confirm rejection and offer to process next candidate.
                """,
    tools=[FunctionTool(human_approve,require_confirmation=True)],
    output_key="final_decision",
    after_agent_callback=auto_save_to_memory,
)

# 7 Summarizer Agent
summarizer = Agent(
    name="Summarizer",
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    instruction="""You are a friendly recruitment summary assistant. Generate a clear, human-readable summary of the candidate evaluation.
                INCLUDE IN YOUR SUMMARY:
                1. Candidate Overview: Name and basic profile
                2. Match Score: Overall score and what it means
                3. Key Strengths: Top 2-3 positive points
                4. Areas of Concern: Any red flags or weaknesses (if any)
                5. Final Recommendation: Proceed/Review/Reject with brief reasoning
                6. Suggested Next Steps: What should happen next
                STYLE GUIDELINES:
                - Write in natural, conversational language
                - Use bullet points for clarity
                - Keep it concise (under 200 words)
                - Never output raw JSON
                """,
    output_key="user_friendly_summary",
    after_agent_callback=auto_save_to_memory,
)

router = Agent(
    name="Router",
    model=Gemini(model=MODEL_NAME,retry_options=retry_config),
    instruction="""You are the main recruitment coordinator and router. Analyze user requests and delegate to the appropriate specialized agent.
                FIRST STEP - ALWAYS:
                Use `load_memory` to check for existing session context (previous JDs, candidates, etc.)
                ROUTING RULES:
                1. **Job Description Creation** → Use `jd_creater`
                - User expresses hiring intent: "I want to hire...", "We need a..."
                - User wants to create/update job requirements
                - NO resume text is present in the message
                
                2. **Resume Screening** → Use `resume_checker`
                - User provides resume text (contains Name, Email, Experience, Education sections)
                - User asks to evaluate or screen a candidate
                
                3. **Summary Generation** → Use `summarizer`
                - After completing a screening workflow
                - When user asks for a summary of results
                - Before presenting final results to user
                
                4. **General Conversation**
                - Greetings, questions about the system, or unclear requests
                - Respond directly without delegating
                CRITICAL RULES:
                - "I want to hire X" WITHOUT resume = `jd_creater` (NOT screener)
                - Never return raw JSON to the user
                - Always use `summarizer` before presenting final evaluation results
                """,
    output_key="routing_decision",
    tools=[
        AgentTool(jd_creater),
        AgentTool(resume_checker),
        AgentTool(summarizer),
        load_memory,
    ]
        
)

recruitment_app = App(
    name="agents",
    root_agent=router,
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=events_compact_config,
    context_cache_config=context_cache_config,
)

runner = Runner(
    app=recruitment_app,
    session_service=session_service,
    memory_service=memory_service,
    # plugins=[
    #     LoggingPlugin()
    # ],
)

async def main():
    response = await runner.run_debug("I want to hire someone")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())