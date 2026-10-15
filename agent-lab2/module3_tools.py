import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()


def lookup_cve_details(cve_id: str) -> str:
    """Look up the CVSS severity score and technical summary for a given CVE identifier.

    Args:
        cve_id: The official CVE string, for example 'CVE-2024-3094'.
    """
    mock_db = {
        "CVE-2024-3094": {
            "cvss": 10.0,
            "affected": "xz-utils 5.6.0 and 5.6.1",
            "summary": "Malicious backdoor injected into liblzma build process hooking RSA_public_decrypt in sshd.",
        },
        "CVE-2021-44228": {
            "cvss": 10.0,
            "affected": "Apache Log4j2 2.0-beta9 to 2.14.1",
            "summary": "Unauthenticated remote code execution via JNDI lookup in logged messages (Log4Shell).",
        },
    }
    record = mock_db.get(cve_id.upper(), {"error": f"No record found for {cve_id}"})
    return json.dumps(record)


def run_manual_tool_roundtrip(user_prompt: str):
    config = types.GenerateContentConfig(
        system_instruction="You are a cybersecurity assistant. Always use tools to verify CVE data.",
        tools=[lookup_cve_details],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.0,
    )

    messages = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=messages,
        config=config,
    )

    if response.function_calls:
        tool_call = response.function_calls[0]
        print(f"[LLM Requested Tool] Name: {tool_call.name} | Args: {tool_call.args}")

        tool_output = lookup_cve_details(**tool_call.args)
        print(f"[Local Function Output] {tool_output}")

        messages.append(response.candidates[0].content)

        function_response_part = types.Part.from_function_response(
            name=tool_call.name,
            response={"result": tool_output}
        )
        messages.append(
            types.Content(role="user", parts=[function_response_part])
        )

        final_res = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=messages,
            config=config,
        )

        print(f"\n[Final LLM Answer]\n{final_res.text}")
    else:
        print(f"[Direct LLM Answer]\n{response.text}")


if __name__ == "__main__":
    run_manual_tool_roundtrip("What is CVE-2024-3094, what versions are affected, and how severe is it?")