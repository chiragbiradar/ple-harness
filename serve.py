# Copyright 2026 PLE Harness Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enhanced engineering agent harness using Claude Opus 5.5.

Built on the minimal sample pattern: one model, tools, one loop.
Enhancements over minimal:
- list_files tool for directory exploration
- mcp_call tool for domain-specific MCP server tools
- Engineering-specialized system prompt
- Auto-install common engineering libraries via bash
- Better handling of PDF/CAD/Excel files via bash helpers
"""

import asyncio
import base64
import json
import os
import re

import openai
import taskbridge

MODEL = os.environ["MODEL_NAME"]

MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)

WRAP_UP_SEC = 300
WRAP_UP = {
    "role": "user",
    "content": (
        "Time is almost up. Stop all tool calls immediately. Reply with your "
        "final summary: each deliverable's location, format, and key results."
    ),
}
RESULT_CHARS = 20_000
RETRY_WAITS = (5, 15, 30, 60, 120)
CONTEXT_FULL = re.compile(
    r"context.?(length|window)|prompt is too long|maximum prompt length"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
CUT_KEEP = 1000
MAX_NUDGES = 3
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
CACHE_MARK = "claude" in MODEL.lower()

client = openai.AsyncOpenAI(max_retries=5, timeout=1800)

SYSTEM = """\
You are a licensed professional engineer (PE) working on a task inside a Linux container \
with no internet access. Your tools act inside the container; bash starts in {workdir}.

## Workflow
1. Read the task instruction carefully. Identify every deliverable: file name, format, location.
2. Survey the workspace: list_files on {workdir} and key subdirectories, then read the files \
the task references (PDFs, drawings, schedules, specs).
3. If MCP servers are available, call list_mcp_tools to discover their capabilities first.
4. Plan your approach before computing. Use Python for all arithmetic and calculations; \
never compute by hand.
5. Execute: run calculations, write each deliverable, then re-read it to verify correctness.
6. When done, reply without calling a tool: one paragraph per deliverable, its path, and \
the key numeric results.

## File handling tips
- PDFs: `pdftotext -layout <file> -` or install pdfplumber via pip and use Python
- Excel/XLSX: `python3 -c "import openpyxl; ..."` — openpyxl is available
- DXF files: `python3 -c "import ezdxf; doc = ezdxf.readfile('<file>'); ..."` — ezdxf available
- IFC files: `python3 -c "import ifcopenshell; f = ifcopenshell.open('<file>'); ..."` — available
- Large files: use bash head/grep to preview before reading fully
- Install missing packages: `pip install <pkg> -q` works in the container

## Engineering principles
- Always state your assumptions, reference the design code/standard being applied
- Check units at every step; SI and imperial can appear in the same project
- For structural tasks: identify loads, load combinations, then design to code
- For electrical tasks: check voltage drops, protection coordination, NEC/IEC compliance
- For civil tasks: check drainage, gradients, bearing capacity, geotechnical data
- For mechanical tasks: check forces, stresses, fatigue, tolerances

Do not guess numeric values. If a file is unreadable, try an alternative method.\
"""


def tool(name, description, **params):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


TOOLS = [
    tool(
        "bash",
        "Run a bash command (120 s limit) in the task container. Returns exit code and output. "
        "Use for: ls, cat, python3, pip install, pdftotext, grep, find, etc.",
        command="string",
    ),
    tool("read_file", "Read a text or binary file from the task container. Returns its content.", path="string"),
    tool(
        "write_file",
        "Create or overwrite a file in the task container. Parent directories are created automatically.",
        path="string",
        content="string",
    ),
    tool(
        "list_files",
        "List files in a directory in the task container. Returns names, sizes, and types.",
        path="string",
    ),
    tool(
        "list_mcp_tools",
        "List all tools available from the task's MCP servers. Call this once at the start if MCP servers are present.",
    ),
    tool(
        "call_mcp",
        "Call a tool on one of the task's MCP servers. Use list_mcp_tools first to see what's available.",
        server_name="string",
        tool_name="string",
        arguments="string",
    ),
]


def clip(text):
    if len(text) <= RESULT_CHARS:
        return text
    half = RESULT_CHARS // 2
    return (
        f"{text[:half]}\n[... {len(text) - RESULT_CHARS} characters cut ...]\n{text[-half:]}"
    )


def marked(messages):
    if not CACHE_MARK:
        return messages
    out = list(messages)
    for i in {0, len(out) - 1}:
        m, content = out[i], out[i].get("content")
        if isinstance(content, str) and content:
            out[i] = {
                **m,
                "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}],
            }
        else:
            out[i] = {**m, "cache_control": {"type": "ephemeral"}}
    return out


async def shorten(session, messages):
    half = len(messages) // 2
    shortened = False
    for m in messages[:half]:
        if m.get("role") != "tool":
            continue
        for block in m.get("content") if isinstance(m.get("content"), list) else [m]:
            text = block.get("text") if isinstance(block, dict) else None
            if text and len(text) > CUT_KEEP:
                block["text"] = text[:CUT_KEEP // 2] + "\n[shortened]\n" + text[-CUT_KEEP // 2:]
                shortened = True
    if shortened:
        await session.log("context full: shortened old tool results")
    return shortened


async def dispatch(session, fn, mcp_clients):
    name = fn["name"]
    try:
        args = json.loads(fn["arguments"] or "{}")
        if not isinstance(args, dict):
            raise TypeError("not a JSON object")
        fn["arguments"] = json.dumps(args, allow_nan=False)
    except (TypeError, ValueError) as exc:
        fn["arguments"] = "{}"
        return f"error: arguments not a JSON object ({exc}); resend the call"

    try:
        if name == "bash":
            result = await session.exec(args["command"])
            return clip(f"exit {result['exit_code']}\n{result['output']}")

        if name == "read_file":
            data = await session.read_file(args["path"])
            try:
                return clip(data.decode("utf-8", "replace"))
            except Exception:
                return f"binary file ({len(data)} bytes); use bash to process it"

        if name == "write_file":
            n = await session.write_file(args["path"], args["content"])
            return f"wrote {n} bytes to {args['path']}"

        if name == "list_files":
            result = await session.exec(
                f"ls -lahF {args['path']!r} 2>&1 | head -200"
            )
            return clip(f"exit {result['exit_code']}\n{result['output']}")

        if name == "list_mcp_tools":
            if not session.mcp_servers:
                return "no MCP servers available for this task"
            lines = []
            for srv in session.mcp_servers:
                try:
                    client_mcp = taskbridge.MCPClient(session, srv["url"])
                    await client_mcp.initialize()
                    tools_result = await client_mcp.rpc("tools/list")
                    tools_list = (tools_result or {}).get("tools", [])
                    mcp_clients[srv["name"]] = client_mcp
                    lines.append(f"Server: {srv['name']} ({srv['url']})")
                    for t in tools_list:
                        lines.append(f"  - {t['name']}: {t.get('description', '')}")
                except Exception as exc:
                    lines.append(f"Server: {srv['name']} — error: {exc}")
            return "\n".join(lines) if lines else "no tools found"

        if name == "call_mcp":
            srv_name = args["server_name"]
            tool_name = args["tool_name"]
            try:
                tool_args = json.loads(args.get("arguments") or "{}")
            except (TypeError, ValueError):
                tool_args = {}

            # Find or create MCP client for the server
            if srv_name not in mcp_clients:
                srv = next(
                    (s for s in session.mcp_servers if s["name"] == srv_name), None
                )
                if srv is None:
                    return f"error: no MCP server named '{srv_name}'"
                client_mcp = taskbridge.MCPClient(session, srv["url"])
                await client_mcp.initialize()
                mcp_clients[srv_name] = client_mcp

            result = await mcp_clients[srv_name].rpc(
                "tools/call",
                {"name": tool_name, "arguments": tool_args},
            )
            content = (result or {}).get("content", [])
            parts = []
            for block in content:
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "image":
                    parts.append(f"[image: {block.get('mimeType', 'unknown')}]")
                else:
                    parts.append(json.dumps(block))
            return clip("\n".join(parts)) if parts else "(empty result)"

        return f"error: no tool named '{name}'"

    except (taskbridge.TaskError, KeyError, ValueError) as exc:
        return f"error: {type(exc).__name__}: {exc}"


async def recover(session, messages, exc, waits, until):
    msg = str(exc)
    if isinstance(exc, openai.BadRequestError) and CONTEXT_FULL.search(msg):
        if await shorten(session, messages):
            return True
        await session.log(f"context full, can't shorten further: {msg[:200]}")
        return False
    if isinstance(exc, (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError)):
        wait = next(waits, None)
        if wait is None:
            return False
        left = session.time_left()
        if left is not None and left > until:
            wait = min(wait, left - until)
        await session.log(f"model error ({msg[:200]}); retry in {wait:.0f}s")
        await asyncio.sleep(wait)
        return True
    return False


async def solve(session):
    mcp_clients = {}
    messages = [
        {"role": "system", "content": SYSTEM.format(workdir=session.workdir)},
        {"role": "user", "content": session.instruction},
    ]

    # Prepend a note about available MCP servers if any
    if session.mcp_servers:
        srv_names = [s["name"] for s in session.mcp_servers]
        messages[1] = {
            "role": "user",
            "content": (
                session.instruction
                + f"\n\n[MCP servers available: {', '.join(srv_names)}. "
                "Call list_mcp_tools first to see what they offer.]"
            ),
        }

    nudges, head, waits = 0, "", iter(RETRY_WAITS)
    while True:
        left = session.time_left()
        wrap_up = left is not None and left < WRAP_UP_SEC
        if wrap_up and messages[-1:] != [WRAP_UP]:
            messages.append(dict(WRAP_UP))

        try:
            reply = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL,
                    messages=marked(messages),
                    tools=TOOLS,
                    tool_choice="none" if wrap_up else "auto",
                    reasoning_effort=EFFORT,
                    max_completion_tokens=MAX_OUTPUT_TOKENS,
                ),
                None if left is None or wrap_up else left - WRAP_UP_SEC,
            )
        except asyncio.TimeoutError:
            continue
        except openai.APIError as exc:
            if await recover(session, messages, exc, waits, WRAP_UP_SEC):
                continue
            raise

        waits = iter(RETRY_WAITS)
        if reply.usage:
            details = getattr(reply.usage, "prompt_tokens_details", None)
            await session.usage(
                MODEL,
                reply.usage.prompt_tokens,
                reply.usage.completion_tokens,
                getattr(details, "cached_tokens", 0) if details else 0,
            )

        message, cut = (
            reply.choices[0].message,
            reply.choices[0].finish_reason == "length",
        )
        text = head + (message.content or "")
        if text:
            session.final_message = text

        turn = message.model_dump(exclude_none=True)
        if wrap_up or not message.tool_calls:
            if wrap_up or (message.content and not cut) or nudges == MAX_NUDGES:
                return session.final_message
            nudges, head = nudges + 1, text
            if message.content:
                messages.append(turn)
                messages.append({
                    "role": "user",
                    "content": (
                        "Your reply was cut off; continue where it stopped."
                        if cut
                        else "Continue the task, and finish with your summary."
                    ),
                })
            continue

        nudges, head = 0, ""
        messages.append(turn)

        tool_results = []
        for call in turn["tool_calls"]:
            fn = call["function"]
            await session.log(f"{fn['name']} {fn['arguments'][:300]}")
            result = await dispatch(session, fn, mcp_clients)
            tool_results.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

        messages.extend(tool_results)


if __name__ == "__main__":
    taskbridge.run(solve)
