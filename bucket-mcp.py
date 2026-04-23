#!/usr/bin/env python3
"""
bucket-mcp.py — Bucket Foundation MCP server (stdio, JSON-RPC 2.0)

Zero-dependency stdio MCP server exposing bucket.foundation research, citation,
and canon tools to any MCP client (Claude Desktop, Claude Code, etc).

Registration (one line, user scope):
  claude mcp add --scope user --transport stdio bucket -- \
    bash -lc "exec python3 /home/gian/agfarms/bucket-mcp/bucket-mcp.py"

Then restart Claude Code. Three tools become available everywhere:
  • bucket_research(query, tier?)   — queries bucket.foundation/api/research
                                      (zero-key proxy to feed402 rail)
  • bucket_cite(doi_or_url)         — returns a CSL-JSON citation block
  • bucket_canon_list(branch?)      — lists canon-tier entries

Protocol: MCP over stdio — newline-delimited JSON-RPC 2.0 on stdin/stdout.
All logging goes to stderr. Never print anything to stdout except JSON-RPC.
"""

from __future__ import annotations

import json
import os
import sys
import re
import traceback
import urllib.request
import urllib.parse
import urllib.error
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "bucket"
SERVER_VERSION = "0.1.0"

BUCKET_BASE = os.environ.get("BUCKET_BASE_URL", "https://www.bucket.foundation")
RESEARCH_PATH = "/api/research"
CANON_PATH = "/api/canon"
HTTP_TIMEOUT = 30.0


def log(msg: str) -> None:
    print(f"[{SERVER_NAME}] {msg}", file=sys.stderr, flush=True)


def http_json(method: str, url: str, body: Any = None) -> dict:
    data = None
    headers = {"Accept": "application/json", "User-Agent": f"bucket-mcp/{SERVER_VERSION}"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return {"ok": True, "status": resp.status, "body": json.loads(raw)}
            except json.JSONDecodeError:
                return {"ok": True, "status": resp.status, "body": raw}
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            raw = str(e)
        return {"ok": False, "status": e.code, "error": raw}
    except Exception as e:
        return {"ok": False, "status": 0, "error": f"{type(e).__name__}: {e}"}


# ---------- Tool implementations ----------

def tool_bucket_research(args: dict) -> dict:
    query = args.get("query")
    tier = args.get("tier", "insight")
    if not query or not isinstance(query, str):
        raise ValueError("query (string) is required")
    if tier not in ("raw", "query", "insight"):
        raise ValueError("tier must be one of: raw, query, insight")
    url = BUCKET_BASE.rstrip("/") + RESEARCH_PATH
    result = http_json("POST", url, {"query": query, "tier": tier})
    if not result["ok"]:
        # Fall back to a helpful error the agent can render
        return {
            "ok": False,
            "error": result.get("error"),
            "hint": (
                "bucket.foundation /api/research is not reachable or not "
                "deployed yet. The proxy is shipping on Track A. Once live, "
                "this tool returns a cited envelope (feed402/0.2) with no "
                "API key required on the MCP user's side."
            ),
            "upstream_status": result.get("status"),
        }
    return {"ok": True, "tier": tier, "query": query, "envelope": result["body"]}


DOI_RE = re.compile(r"10\.\d{4,9}/[^\s]+", re.IGNORECASE)


def tool_bucket_cite(args: dict) -> dict:
    ref = args.get("doi_or_url")
    if not ref or not isinstance(ref, str):
        raise ValueError("doi_or_url (string) is required")

    # Normalize: if URL contains a DOI, extract it
    doi_match = DOI_RE.search(ref)
    if doi_match:
        doi = doi_match.group(0).rstrip(".,;)")
        # Call doi.org content negotiation for CSL-JSON
        url = f"https://doi.org/{doi}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.citationstyles.csl+json",
                "User-Agent": f"bucket-mcp/{SERVER_VERSION}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                csl = json.loads(resp.read().decode("utf-8"))
            return {"ok": True, "doi": doi, "csl_json": csl}
        except Exception as e:
            return {
                "ok": False,
                "doi": doi,
                "error": f"doi.org content negotiation failed: {type(e).__name__}: {e}",
            }

    # Non-DOI URL → minimal webpage CSL stub
    return {
        "ok": True,
        "csl_json": {
            "type": "webpage",
            "URL": ref,
            "accessed": {"raw": "bucket-mcp"},
            "id": ref,
        },
        "note": "no DOI detected; returned minimal webpage CSL stub",
    }


CANON_BRANCHES = [
    "mathematics",
    "physics",
    "chemistry",
    "information",
    "biophysics",
    "cosmology",
    "mind",
]


def tool_bucket_canon_list(args: dict) -> dict:
    branch = args.get("branch")
    if branch is not None:
        if branch not in CANON_BRANCHES:
            raise ValueError(f"branch must be one of: {', '.join(CANON_BRANCHES)}")
    # Try the live canon endpoint first
    url = BUCKET_BASE.rstrip("/") + CANON_PATH
    if branch:
        url += "?branch=" + urllib.parse.quote(branch)
    result = http_json("GET", url)
    if result["ok"]:
        return {"ok": True, "branch": branch, "index": result["body"]}
    # Fallback: static branch listing
    return {
        "ok": True,
        "branch": branch,
        "branches": CANON_BRANCHES,
        "note": (
            "Live /api/canon endpoint not reachable; returning static branch "
            "index. Canon artifacts live at "
            "gdrive:AGFarms/Nucleus/research/bucket-canon/ and are mirrored "
            "from the Viatika x402 research pipeline."
        ),
        "upstream_status": result.get("status"),
    }


# ---------- MCP wiring ----------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "bucket_research",
        "description": (
            "Query the Bucket Foundation research rail. Routes through "
            "bucket.foundation/api/research (zero-key proxy) → feed402 "
            "provider → cited envelope. Default tier 'insight' is cheapest "
            "and best for agent traffic. 'raw' returns full rows, 'query' "
            "returns structured top-k."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language research question."},
                "tier": {
                    "type": "string",
                    "enum": ["raw", "query", "insight"],
                    "default": "insight",
                    "description": "feed402 tier. Default: insight ($0.002/call).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "bucket_cite",
        "description": (
            "Return a CSL-JSON citation block for a DOI or URL. Uses doi.org "
            "content negotiation when a DOI is present."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "doi_or_url": {
                    "type": "string",
                    "description": "A DOI (10.xxxx/...) or a URL that may contain one.",
                }
            },
            "required": ["doi_or_url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "bucket_canon_list",
        "description": (
            "List canon-tier entries. Canon holds only foundations — axioms, "
            "laws, primary derivations — across 7 branches. Omit branch for "
            "all; pass one of: mathematics, physics, chemistry, information, "
            "biophysics, cosmology, mind."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "enum": CANON_BRANCHES,
                    "description": "Optional canon branch filter.",
                }
            },
            "additionalProperties": False,
        },
    },
]

TOOL_HANDLERS = {
    "bucket_research": tool_bucket_research,
    "bucket_cite": tool_bucket_cite,
    "bucket_canon_list": tool_bucket_canon_list,
}


def respond(id_: Any, result: Any = None, error: dict | None = None) -> None:
    msg: dict = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle(req: dict) -> None:
    method = req.get("method")
    id_ = req.get("id")
    params = req.get("params") or {}

    if method == "initialize":
        respond(id_, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
        return

    if method == "notifications/initialized":
        return  # no response for notifications

    if method == "tools/list":
        respond(id_, {"tools": TOOLS})
        return

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            respond(id_, error={"code": -32601, "message": f"unknown tool: {name}"})
            return
        try:
            result = handler(args)
            respond(id_, {
                "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
                "isError": not result.get("ok", True) if isinstance(result, dict) else False,
            })
        except Exception as e:
            log(f"tool {name} raised: {e}\n{traceback.format_exc()}")
            respond(id_, {
                "content": [{"type": "text", "text": f"error: {type(e).__name__}: {e}"}],
                "isError": True,
            })
        return

    if method == "ping":
        respond(id_, {})
        return

    if id_ is not None:
        respond(id_, error={"code": -32601, "message": f"method not found: {method}"})


def main() -> int:
    log(f"starting {SERVER_NAME} v{SERVER_VERSION} (bucket base: {BUCKET_BASE})")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"bad JSON on stdin: {e}")
            continue
        try:
            handle(req)
        except Exception as e:
            log(f"handler crash: {e}\n{traceback.format_exc()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
