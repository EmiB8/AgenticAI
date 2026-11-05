"""agent_core.py - the reusable agent harness you build during Lab 3.

Ex1: complete execute_tool() and the TODOs in run_agent() (the guarded ReAct loop).
Ex2: complete the approval-policy TODO in run_agent().
Ex3: reuse run_agent() unchanged for the orchestrator.
"""
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
client = genai.Client()
MODEL = os.getenv("LAB_MODEL", "gemini-3.5-flash-lite")  # fallback: gemini-3.1-flash-lite


@dataclass
class AgentResult:
    answer: str
    stop_reason: str  # "done" | "max_steps" | "stuck" | "provider_error"
    steps: int
    prompt_tokens_per_step: list[int] = field(default_factory=list)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(self.prompt_tokens_per_step)


@dataclass
class Decision:
    """Ex2: what the human operator decided about a risky tool call."""
    approved: bool
    args: dict | None = None  # set when the operator EDITED the arguments
    feedback: str = ""  # free-text reason, sent back to the model


Approver = Callable[[str, dict], Decision]


def call_llm(contents, config, retries: int = 5):
    """One model call, with retry + exponential backoff on 429/5xx (given)."""
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=MODEL, contents=contents, config=config)
        except errors.APIError as e:
            if e.code in (429, 500, 503) and attempt < retries - 1:
                wait = 2 ** attempt * 5
                print(f"  .. provider error {e.code}, retrying in {wait}s")
                time.sleep(wait)
            else:
                raise


def execute_tool(tools_map: dict[str, Callable], name: str, args: dict) -> str:
    """Gate: known tool? then run it. Every failure must become a readable string, never a crash."""
    if name not in tools_map:
        return f"ERROR: Unknown tool '{name}'. Available tools: {', '.join(tools_map.keys())}."
    try:
        fn = tools_map[name]
        return str(fn(**args))
    except TypeError as e:
        return f"ERROR: Bad arguments for {name}: {e}"
    except Exception as e:
        return f"ERROR: {name} failed: {e}"


def run_agent(
        user_prompt: str,
        tools: list[Callable],
        system_instruction: str,
        max_steps: int = 8,
        requires_approval: frozenset[str] = frozenset(),
        approver: Approver | None = None,
        show_thoughts: bool = False,
        verbose: bool = True,
) -> AgentResult:
    tools_map = {fn.__name__: fn for fn in tools}
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        # flash-lite thinks minimally by default; "high" + include_thoughts returns a readable summary.
        thinking_config=(types.ThinkingConfig(include_thoughts=True, thinking_level="high")
                         if show_thoughts else None),
        # No temperature override: Google recommends the default (1.0) for Gemini 3 models.
    )
    messages = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])]
    tokens: list[int] = []
    seen: Counter = Counter()

    for step in range(1, max_steps + 1):
        try:
            response = call_llm(messages, config)
        except errors.APIError as e:  # abnormal stop: report it, don't crash
            return AgentResult(f"STOPPED: provider error {e.code}: {e.message}", "provider_error", step, tokens)

        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        tokens.append(prompt_tokens)
        if verbose:
            print(f"\n--- [Step {step}/{max_steps}] input tokens: {prompt_tokens:,} ---")
        content = response.candidates[0].content
        if verbose and content and content.parts:
            for part in content.parts:
                if part.thought and part.text:
                    print(f"  [Thought] {part.text.strip()[:300]}")

        # Stop condition A: no tool calls -> final answer
        if not response.function_calls:
            return AgentResult(response.text or "", "done", step, tokens)

        # 1. Append the model turn AS-IS (it carries Gemini 3 thought signatures)
        messages.append(content)

        parts = []
        for call in response.function_calls:
            args = dict(call.args or {})
            if verbose:
                print(f"  -> [Tool Call] {call.name}({json.dumps(args)[:200]})")

            # STRETCH (Ex1): stuck-agent detector.
            call_signature = (call.name, tuple(sorted(args.items())))
            seen[call_signature] += 1

            if seen[call_signature] >= 3:
                return AgentResult(
                    f"STOPPED: Agent got stuck repeatedly calling {call.name} with identical arguments.",
                    "stuck", step, tokens
                )
            elif seen[call_signature] == 2:
                output = f"NOTE: Repeated call to {call.name} with identical arguments. Use the earlier result."
            else:
                # TODO (Ex2): policy layer.
                if call.name in requires_approval:
                    if approver is None:
                        output = f"ERROR: {call.name} requires approval, but no approver was provided."
                    else:
                        decision = approver(call.name, args)
                        if not decision.approved:
                            output = f"DENIED BY OPERATOR: {decision.feedback}. Adjust your plan."
                        else:
                            if decision.args is not None:
                                args = decision.args
                                tool_result = execute_tool(tools_map, call.name, args)
                                output = f"NOTE: Operator approved but EDITED arguments to {args}.\nResult: {tool_result}"
                            else:
                                output = execute_tool(tools_map, call.name, args)
                else:
                    output = execute_tool(tools_map, call.name, args)

            if verbose:
                print(f"  <- [Tool Output] {str(output)[:160]}")

                # TODO (Ex1): append ONE function_response part per call to `parts`
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call.id,
                            name=call.name,
                            response={"result": output}
                        )
                    )
                )

        # TODO (Ex1): append ALL parts in ONE message. Gemini has no "tool" role.
        messages.append(types.Content(role="user", parts=parts))

    # Stop condition B: budget exhausted -> partial progress, not a crash
    return AgentResult(
        f"STOPPED after {max_steps} steps without a final answer. Raise max_steps or narrow the task.",
        "max_steps", max_steps, tokens,
    )
