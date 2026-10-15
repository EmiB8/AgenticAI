from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

# TODO: Add the missing fields to LogAnalysis with clear Field(description="...") hints:
# - severity: SeverityLevel
# - error_code: int
# - is_compromised: bool
# - recommended_fix: list[str]
class LogAnalysis(BaseModel):
    summary: str = Field(description="One-sentence technical summary of the event")
    severity: SeverityLevel = Field(description="Severity level of the event")
    error_code: int = Field(description="Error code of the event")
    is_compromised: bool = Field(description="Wether the system is compromised or not")
    recommended_fix: list[str] = Field(description="List of fixes that are recommended in order to solve the problems")
# TODO: Improve this system prompt so the model accurately classifies SIEM logs
SYSTEM_PROMPT =  """You are a cybersecurity expert analyzing server logs. Your task is to extract key information from the logs and provide a structured analysis.
For each log entry, identify:
- severity level (LOW, MEDIUM, HIGH, CRITICAL)
- error code (if present and what it stands for)
- is_compromised (True if the log indicates so, otherwise False, also provide a brief explanation with the reasoning)
- recommended_fix (a list of up to 5 actinable recommendations to mitigate the issue)
Return the analysis in JSON format adhering to the LogAnalysis schema. Ensure that the output is valid JSON and includes all required fields. If any information is missing, provide a reasonable default or indicate that it is not available."""


def parse_server_log(raw_log: str) -> LogAnalysis:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=f"Analyze the following raw server log:\n{raw_log}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=LogAnalysis,
        ),
    )
    # Validate and deserialize JSON string directly into the Pydantic object
    return LogAnalysis.model_validate_json(response.text)

if __name__ == "__main__":
    sample = (
        "2026-10-15 02:11:44 [ALERT] sshd[9012]: Accepted password for root "
        "from 198.51.100.77 port 44122 ssh2 (ErrorCode=401 bypassed via PAM bug)"
    )
    print(parse_server_log(sample).model_dump_json(indent=2))

# --- Automated Unit Test (Run with: pytest test_module2_structured.py) ---
def test_ssh_breach_log_parsing():
    sample_log = (
        "2026-10-15 02:11:44 [ALERT] sshd[9012]: Accepted password for root "
        "from 198.51.100.77 port 44122 ssh2 (ErrorCode=401 bypassed via PAM bug)"
    )
    result = parse_server_log(sample_log)

    assert result.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
    assert result.error_code == 401
    assert result.is_compromised is True
    assert len(result.recommended_fix) >= 1
