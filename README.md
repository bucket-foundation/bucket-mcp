# bucket-mcp

Zero-dependency stdio MCP server for the **Bucket Foundation** research rail.
Plug into Claude Desktop or Claude Code and get three tools:

| Tool | What |
|------|------|
| `bucket_research(query, tier?)` | Call `bucket.foundation/api/research` (zero-key proxy over the feed402 rail). Default tier `insight` ($0.002/call). |
| `bucket_cite(doi_or_url)` | Return a CSL-JSON citation block. Uses doi.org content negotiation for DOIs. |
| `bucket_canon_list(branch?)` | List canon-tier entries. Seven branches: `mathematics`, `physics`, `chemistry`, `information`, `biophysics`, `cosmology`, `mind`. |

Canon holds only **foundations** — axioms, laws, primary derivations. Outcomes
(longevity, disease, cognition) are downstream applications, not canon.

## Install (Claude Code / Claude Desktop, user scope)

```bash
# 1. Clone
gh repo clone bucket-foundation/bucket-mcp ~/bucket-mcp
# or: git clone https://github.com/bucket-foundation/bucket-mcp ~/bucket-mcp

# 2. Register at user scope (works in every Claude session on this machine)
claude mcp add --scope user --transport stdio bucket -- \
  bash -lc "exec python3 $HOME/bucket-mcp/bucket-mcp.py"

# 3. Restart Claude Code / Desktop
claude mcp list   # should show 'bucket' green
```

No `pip install`, no virtualenv, no API key on the MCP user's side. Python 3.8+
stdlib only.

## Zero-install (planned)

```bash
# Node (once published):
npx bucket-mcp

# Python via uv:
uvx bucket-mcp
```

Both are **pending publish**. Use the git clone path above in the meantime.

## Environment

| Var | Default | Meaning |
|-----|---------|---------|
| `BUCKET_BASE_URL` | `https://www.bucket.foundation` | Bucket site base. Override for staging/local. |

## Architecture

```
Claude agent
  │
  ▼ (stdio JSON-RPC 2.0)
bucket-mcp.py
  │
  ▼ (HTTPS)
bucket.foundation/api/research   ← Track A zero-key proxy
  │
  ▼ (x402 + feed402/0.2)
x402-research.agfarms.dev        ← live reference merchant
  │
  ▼
PubMed / OpenAlex / Semantic Scholar / ClinicalTrials / PubChem / Kruse corpus
```

The MCP user pays nothing. The proxy at bucket.foundation holds the funded
wallet and settles on Base. Agents get cited envelopes; humans get receipts.

## Protocol

MCP over stdio, JSON-RPC 2.0, protocol version `2024-11-05`. All logging on
stderr; only JSON-RPC messages on stdout. Matches the pattern used by the
AGFarms in-house [figma-agf](https://github.com/AGFarms/tools) MCP server.

## License

MIT (code). The feed402 spec it speaks is CC0.

## Part of

[Bucket Foundation](https://www.bucket.foundation) — **build the past. build
history. bucket is the new renaissance.** Reference implementation of an open
protocol for paid-for-once, citeable-forever primary research.
