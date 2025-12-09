"""
NeMo Guardrails Custom Actions
"""
import re
from typing import Optional

# try import nemoguardrails, if failed, provide fallback
try:
    from nemoguardrails.actions import action
except ImportError:
    # Fallback: if nemoguardrails is not installed, provide an empty decorator
    def action(name: Optional[str] = None):
        def decorator(func):
            return func
        return decorator



# Configuration


# blocked words list - can be extended
BLOCKED_WORDS = [
    # security related
    "hack", "exploit", "bypass", "jailbreak",
    "password stealing", "credit card theft",
    # inappropriate content
    "illegal", "violence", "weapon",
    # recruitment discrimination
    "only hire male", "only hire female",
    "no disabled", "age limit",
]

# Prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"override\s+your\s+instructions",
    r"new\s+instructions?:",
]



# Input Guardrail Actions

@action()
async def check_blocked_words(text: str) -> bool:
    if not text:
        return False
    
    text_lower = text.lower()
    for word in BLOCKED_WORDS:
        if word.lower() in text_lower:
            return True
    return False


@action()
async def check_input_length(text: str, max_length: int = 5000) -> bool:
    if not text:
        return False
    return len(text) > max_length


@action()
async def check_prompt_injection(text: str) -> bool:
    if not text:
        return False
    
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False



# Output Guardrail Actions

@action()
async def mask_pii(text: str) -> str:
    if not text:
        return text
    
    # mask email
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '[EMAIL REDACTED]',
        text
    )
    
    # mask phone number (multiple formats)
    text = re.sub(
        r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        '[PHONE REDACTED]',
        text
    )
    
    # mask SSN
    text = re.sub(
        r'\b\d{3}-\d{2}-\d{4}\b',
        '[SSN REDACTED]',
        text
    )
    
    # mask credit card number
    text = re.sub(
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        '[CARD REDACTED]',
        text
    )
    
    # mask IP address
    text = re.sub(
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        '[IP REDACTED]',
        text
    )
    
    return text




