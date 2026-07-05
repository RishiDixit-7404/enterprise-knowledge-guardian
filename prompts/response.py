"""Response agent prompts."""

PROMPT_ID = "response_agent_prompt"
PROMPT_VERSION = "v1"
PROMPT_SOURCE = "prompts/response.py"

RESPONSE_PROMPT_TEMPLATE = """You are an expert Q&A agent.

Question: {question}

Generate a clear, accurate, and professional answer using ONLY the supported claims provided.
Cite the chunk IDs provided in the claims.
"""
