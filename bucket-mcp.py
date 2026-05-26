#!/usr/bin/env python3
"""
bucket-mcp.py — Bucket Foundation MCP server (stdio, JSON-RPC 2.0)

Zero-dependency stdio MCP server exposing bucket.foundation research, citation,
and canon tools to any MCP client (Claude Desktop, Claude Code, etc).

Registration (one line, user scope):
  claude mcp add --scope user --transport stdio bucket -- \
    bash -lc "exec python3 /home/gian/agfarms/bucket-mcp/bucket-mcp.py"

Then restart Claude Code. Five tools become available everywhere:
  • bucket_research(query, tier?)   — queries bucket.foundation/api/research
                                      (zero-key proxy to feed402 rail)
  • bucket_cite(doi_or_url)         — returns a CSL-JSON citation block
  • bucket_canon_list(branch?)      — lists canon-tier entries
  • derbyfish_fingerprint_lookup(image_url? | image_bytes_b64?)
                                    — "Shazam for individual fish": vision-worker
                                      segment+embed → k-NN over the BHRV fish
                                      fingerprint index → match + catch history
  • derbyfish_population_health(water_body, species, window?)
                                    — K-factor distribution, length percentiles,
                                      catch counts + confidence for a water body
                                      × species over a time window

Protocol: MCP over stdio — newline-delimited JSON-RPC 2.0 on stdin/stdout.
All logging goes to stderr. Never print anything to stdout except JSON-RPC.

DerbyFish tool dependencies (read-only, no writes):
  • Vision worker (segment/embed): VISION_WORKER_URL, default http://127.0.0.1:8181.
    Required only by derbyfish_fingerprint_lookup. If unreachable from the MCP
    runtime, the tool returns a structured {ok: false, ...} envelope naming the
    network requirement rather than hard-failing the server.
  • Analytics warehouse (Postgres / pgvector): DERBYFISH_ANALYTICS_HOST (default
    127.0.0.1, expects an SSH tunnel to 5.161.236.151:5433), DERBYFISH_ANALYTICS_PORT
    (default 5433), DERBYFISH_ANALYTICS_DB (default derbyfish_index),
    DERBYFISH_ANALYTICS_USER (default dbt_runner), DERBYFISH_ANALYTICS_PASSWORD.
    psycopg2 is imported lazily so the other three tools work without it installed.
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

# ---------- DerbyFish (BHRV vision-AI) config ----------
# Vision worker: the FastAPI sidecar that owns SAM2 segmentation + SigLIP/DINOv2
# embeddings. Default :8181 matches where it currently runs on the dbt host;
# the IDX module convention elsewhere is :8081 — both are loopback-only.
VISION_WORKER_URL = os.environ.get("VISION_WORKER_URL", "http://127.0.0.1:8181").rstrip("/")
VISION_WORKER_TIMEOUT_S = float(os.environ.get("VISION_WORKER_TIMEOUT_S", "60"))

# Analytics warehouse (read-only). Host defaults to loopback because access is
# normally via an SSH tunnel to 5.161.236.151:5433 (see derbyfish-dbt docs).
DF_DB = {
    "host": os.environ.get("DERBYFISH_ANALYTICS_HOST", "127.0.0.1"),
    "port": int(os.environ.get("DERBYFISH_ANALYTICS_PORT", "5433")),
    "dbname": os.environ.get("DERBYFISH_ANALYTICS_DB", "derbyfish_index"),
    "user": os.environ.get("DERBYFISH_ANALYTICS_USER", "dbt_runner"),
    "password": os.environ.get("DERBYFISH_ANALYTICS_PASSWORD", ""),
}
DF_DB_CONNECT_TIMEOUT_S = int(os.environ.get("DERBYFISH_ANALYTICS_CONNECT_TIMEOUT", "10"))

# SigLIP solo-fish embedding dimensionality (silver.fish_fingerprint.solo_siglip_emb).
SIGLIP_DIM = 1152
# k-NN match acceptance threshold (cosine similarity). Below this we report a
# novel fish rather than a (low-confidence) match.
FINGERPRINT_MATCH_THRESHOLD = float(os.environ.get("DERBYFISH_FINGERPRINT_THRESHOLD", "0.90"))
# Default time window for population.health when the caller omits one.
DEFAULT_POPULATION_WINDOW = os.environ.get("DERBYFISH_POPULATION_WINDOW", "365 days")


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


# ---------- DerbyFish helpers ----------

def _df_db_query(sql: str, params: tuple = ()) -> dict:
    """Run a read-only query against the analytics warehouse.

    Returns {"ok": True, "rows": [dict, ...], "columns": [...]} or
    {"ok": False, "error": ..., "hint": ...}. psycopg2 is imported lazily so the
    three Bucket tools keep working on a host that lacks it.
    """
    try:
        import psycopg2
        import psycopg2.extras
    except Exception as e:  # pragma: no cover - import-environment dependent
        return {
            "ok": False,
            "error": f"psycopg2 not importable: {type(e).__name__}: {e}",
            "hint": "pip install psycopg2-binary in the MCP server's runtime.",
        }
    if not DF_DB["password"]:
        return {
            "ok": False,
            "error": "DERBYFISH_ANALYTICS_PASSWORD is not set",
            "hint": (
                "Export the dbt_runner password (see derbyfish-dbt/.env) so the "
                "MCP server can read the warehouse. Read-only; no writes."
            ),
        }
    conn = None
    try:
        conn = psycopg2.connect(connect_timeout=DF_DB_CONNECT_TIMEOUT_S, **DF_DB)
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()] if cur.description else []
            cols = [d.name for d in cur.description] if cur.description else []
        return {"ok": True, "rows": rows, "columns": cols}
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "hint": (
                f"Warehouse unreachable at {DF_DB['host']}:{DF_DB['port']}/"
                f"{DF_DB['dbname']}. It is normally reached via an SSH tunnel to "
                "5.161.236.151:5433 (see derbyfish-dbt docs §11)."
            ),
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _vector_literal(vec: list) -> str:
    """pgvector text literal: '[f0,f1,...]'."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _vision_post(path: str, body: dict) -> dict:
    """POST to the vision worker; reuse the stdlib http_json helper so we keep
    the zero-extra-dep posture of the existing tools."""
    url = VISION_WORKER_URL + path
    return http_json("POST", url, body)


# ---------- E5.T5: derbyfish.fingerprint.lookup(image) ----------

def tool_derbyfish_fingerprint_lookup(args: dict) -> dict:
    """Shazam for individual fish.

    image_url OR image_bytes_b64 -> /vision/segment (solo_fish) -> /vision/embed
    -> k-NN over silver.fish_fingerprint.solo_siglip_emb (HNSW cosine) -> match
    + catch history, or {novel_fish: true} when nothing clears the threshold.
    """
    image_url = args.get("image_url") or args.get("image")
    image_b64 = args.get("image_bytes_b64")
    if not image_url and not image_b64:
        raise ValueError("provide image_url (string) or image_bytes_b64 (base64 string)")
    threshold = args.get("threshold", FINGERPRINT_MATCH_THRESHOLD)
    top_k = int(args.get("top_k", 5))
    if top_k < 1 or top_k > 25:
        raise ValueError("top_k must be between 1 and 25")

    media_field = {"media_url": image_url} if image_url else {"media_bytes_b64": image_b64}

    # 1) segment the solo fish
    seg = _vision_post("/vision/segment", {**media_field, "frame_tag": "solo_fish"})
    if not seg["ok"]:
        return {
            "ok": False,
            "stage": "vision/segment",
            "error": seg.get("error"),
            "upstream_status": seg.get("status"),
            "hint": (
                f"Vision worker not reachable at {VISION_WORKER_URL}. This tool "
                "needs loopback (or VISION_WORKER_URL) access to the BHRV vision "
                "sidecar that runs the SAM2 + SigLIP models. Run the MCP server "
                "on the dbt host, or set VISION_WORKER_URL to a reachable worker."
            ),
        }
    seg_body = seg["body"] if isinstance(seg.get("body"), dict) else {}
    mask_url = seg_body.get("mask_url")
    mask_confidence = seg_body.get("mask_confidence")

    # 2) embed the masked solo fish (SigLIP only — that is the index backbone)
    embed_req = {**media_field, "frame_tag": "solo_fish", "backbones": ["siglip"]}
    if mask_url:
        embed_req["mask_url"] = mask_url
    emb = _vision_post("/vision/embed", embed_req)
    if not emb["ok"]:
        return {
            "ok": False,
            "stage": "vision/embed",
            "error": emb.get("error"),
            "upstream_status": emb.get("status"),
            "hint": f"Vision worker /vision/embed failed at {VISION_WORKER_URL}.",
        }
    emb_body = emb["body"] if isinstance(emb.get("body"), dict) else {}
    siglip = emb_body.get("siglip")
    if not siglip or len(siglip) != SIGLIP_DIM:
        return {
            "ok": False,
            "stage": "vision/embed",
            "error": (
                f"worker returned siglip of length {0 if not siglip else len(siglip)}; "
                f"expected {SIGLIP_DIM}"
            ),
        }

    # 3) k-NN over the populated fingerprint index, joined to catch facts so a
    #    match comes back with its full catch history.
    qvec = _vector_literal(siglip)
    sql = """
        WITH ranked AS (
            SELECT
                ff.submission_id,
                1 - (ff.solo_siglip_emb <=> %s::vector) AS cosine,
                ff.measured_length_cm,
                ff.k_factor
            FROM silver.fish_fingerprint ff
            WHERE ff.solo_siglip_emb IS NOT NULL
            ORDER BY ff.solo_siglip_emb <=> %s::vector
            LIMIT %s
        )
        SELECT
            r.submission_id,
            r.cosine,
            COALESCE(r.measured_length_cm, gcf.length_cm) AS length_cm,
            r.k_factor,
            gcf.angler_username,
            gcf.display_name,
            gcf.species_name,
            gcf.bow_name AS water_body,
            gcf.catch_date
        FROM ranked r
        LEFT JOIN gold.gold_catch_facts gcf ON gcf.id = r.submission_id
        ORDER BY r.cosine DESC
    """
    res = _df_db_query(sql, (qvec, qvec, top_k))
    if not res["ok"]:
        return {
            "ok": False,
            "stage": "warehouse",
            "error": res.get("error"),
            "hint": res.get("hint"),
            "mask_confidence": mask_confidence,
        }

    neighbors = res["rows"]
    if not neighbors:
        return {
            "ok": True,
            "novel_fish": True,
            "fingerprint_match": None,
            "reason": "fingerprint index is empty",
            "mask_confidence": mask_confidence,
        }

    best = neighbors[0]
    best_cosine = float(best["cosine"]) if best["cosine"] is not None else 0.0
    if best_cosine < threshold:
        return {
            "ok": True,
            "novel_fish": True,
            "fingerprint_match": None,
            "confidence": round(best_cosine, 4),
            "threshold": threshold,
            "nearest_below_threshold": {
                "submission_id": str(best["submission_id"]),
                "species": best.get("species_name"),
                "cosine": round(best_cosine, 4),
            },
            "mask_confidence": mask_confidence,
        }

    # A "fish" is a fingerprint cluster; here one submission == one fingerprint.
    # The match's catch history = every neighbor sharing that submission's identity.
    # With the current per-catch fingerprint table, history is the matched row
    # plus any other rows that clear the threshold for the same angler/water.
    catch_history = []
    for n in neighbors:
        c = float(n["cosine"]) if n["cosine"] is not None else 0.0
        if c < threshold:
            continue
        catch_history.append({
            "submission_id": str(n["submission_id"]),
            "angler": n.get("angler_username") or n.get("display_name"),
            "date": (n["catch_date"].isoformat() if n.get("catch_date") else None),
            "length_cm": (float(n["length_cm"]) if n.get("length_cm") is not None else None),
            "water_body": n.get("water_body"),
            "k_factor": (float(n["k_factor"]) if n.get("k_factor") is not None else None),
            "cosine": round(c, 4),
        })

    return {
        "ok": True,
        "novel_fish": False,
        "fingerprint_match": str(best["submission_id"]),
        "confidence": round(best_cosine, 4),
        "threshold": threshold,
        "species": best.get("species_name"),
        "mask_confidence": mask_confidence,
        "catch_history": catch_history,
        "note": (
            "k_factor is NULL until the InstantMesh weight model ships; length and "
            "catch metadata are live. fingerprint_match is the per-catch "
            "submission_id of the nearest verified fish."
        ),
    }


# ---------- E5.T6: derbyfish.population.health(water_body, species, window) ----------

def tool_derbyfish_population_health(args: dict) -> dict:
    """Population-health snapshot for a water body x species over a window.

    Resolves water_body by bow_id (uuid) or name (ILIKE), species by name
    (ILIKE), filters catch_date to the window, and returns length percentiles,
    catch counts, distinct anglers, K-factor stats (pending until weights ship),
    and a sample-size confidence band. Read-only.
    """
    water_body = args.get("water_body")
    species = args.get("species")
    window = args.get("window") or DEFAULT_POPULATION_WINDOW
    if not water_body or not isinstance(water_body, str):
        raise ValueError("water_body (string: bow_id uuid or water-body name) is required")
    if not species or not isinstance(species, str):
        raise ValueError("species (string: species name) is required")

    # bow_id (uuid) vs name. uuid form -> exact match on bow_id, else name ILIKE.
    is_uuid = bool(re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        water_body.strip(),
    ))
    wb_clause = "gcf.bow_id = %s::uuid" if is_uuid else "gcf.bow_name ILIKE %s"
    wb_param = water_body.strip() if is_uuid else f"%{water_body.strip()}%"
    sp_param = f"%{species.strip()}%"

    sql = f"""
        WITH pop AS (
            SELECT
                gcf.id AS submission_id,
                gcf.profile_id,
                gcf.length_cm,
                gcf.bow_name,
                gcf.bow_id,
                gcf.species_name,
                gcf.catch_date,
                ff.k_factor,
                ff.estimated_weight_g
            FROM gold.gold_catch_facts gcf
            LEFT JOIN silver.fish_fingerprint ff ON ff.submission_id = gcf.id
            WHERE {wb_clause}
              AND gcf.species_name ILIKE %s
              AND gcf.catch_date >= (CURRENT_DATE - %s::interval)
        )
        SELECT
            count(*)                                              AS sample_size,
            count(DISTINCT profile_id)                            AS distinct_anglers,
            count(DISTINCT bow_id)                                AS distinct_water_bodies,
            min(catch_date)                                       AS first_catch,
            max(catch_date)                                       AS last_catch,
            count(length_cm)                                      AS n_length,
            round(min(length_cm)::numeric, 1)                     AS len_min_cm,
            round(percentile_cont(0.10) WITHIN GROUP (ORDER BY length_cm)::numeric, 1) AS len_p10_cm,
            round(percentile_cont(0.25) WITHIN GROUP (ORDER BY length_cm)::numeric, 1) AS len_p25_cm,
            round(percentile_cont(0.50) WITHIN GROUP (ORDER BY length_cm)::numeric, 1) AS len_p50_cm,
            round(percentile_cont(0.75) WITHIN GROUP (ORDER BY length_cm)::numeric, 1) AS len_p75_cm,
            round(percentile_cont(0.90) WITHIN GROUP (ORDER BY length_cm)::numeric, 1) AS len_p90_cm,
            round(max(length_cm)::numeric, 1)                     AS len_max_cm,
            round(avg(length_cm)::numeric, 1)                     AS len_mean_cm,
            count(k_factor)                                       AS n_kfactor,
            round(avg(k_factor)::numeric, 3)                      AS k_factor_mean,
            round(percentile_cont(0.50) WITHIN GROUP (ORDER BY k_factor)::numeric, 3) AS k_factor_p50,
            min(bow_name)                                         AS resolved_water_body
        FROM pop
    """
    res = _df_db_query(sql, (wb_param, sp_param, window))
    if not res["ok"]:
        return {"ok": False, "stage": "warehouse", "error": res.get("error"), "hint": res.get("hint")}

    row = res["rows"][0] if res["rows"] else {}
    sample_size = int(row.get("sample_size") or 0)

    if sample_size == 0:
        return {
            "ok": True,
            "water_body": water_body,
            "species": species,
            "window": window,
            "sample_size": 0,
            "confidence": "none",
            "narrative_hint": (
                f"No verified catches of '{species}' at '{water_body}' in the last "
                f"{window}. Either the water body / species name did not match or "
                "the window is too short. Try a broader name or a longer window."
            ),
        }

    # Sample-size -> qualitative confidence band (small dataset; honest about it).
    if sample_size >= 100:
        confidence = "high"
    elif sample_size >= 30:
        confidence = "moderate"
    elif sample_size >= 10:
        confidence = "low"
    else:
        confidence = "very_low"

    n_kfactor = int(row.get("n_kfactor") or 0)
    if n_kfactor == 0:
        k_factor_block = {
            "status": "pending_weight_data",
            "n_kfactor": 0,
            "k_factor_mean": None,
            "k_factor_p50": None,
            "note": (
                "K-factor (100*W/L^3) requires estimated weight, which is NULL "
                "until the InstantMesh 3D weight model ships. Length percentiles "
                "and catch counts below are live."
            ),
        }
    else:
        k_factor_block = {
            "status": "available",
            "n_kfactor": n_kfactor,
            "k_factor_mean": (float(row["k_factor_mean"]) if row.get("k_factor_mean") is not None else None),
            "k_factor_p50": (float(row["k_factor_p50"]) if row.get("k_factor_p50") is not None else None),
        }

    def _f(key):
        v = row.get(key)
        return float(v) if v is not None else None

    return {
        "ok": True,
        "water_body": water_body,
        "resolved_water_body": row.get("resolved_water_body"),
        "species": species,
        "window": window,
        "sample_size": sample_size,
        "distinct_anglers": int(row.get("distinct_anglers") or 0),
        "distinct_water_bodies": int(row.get("distinct_water_bodies") or 0),
        "first_catch": (row["first_catch"].isoformat() if row.get("first_catch") else None),
        "last_catch": (row["last_catch"].isoformat() if row.get("last_catch") else None),
        "confidence": confidence,
        "length_percentiles_cm": {
            "min": _f("len_min_cm"),
            "p10": _f("len_p10_cm"),
            "p25": _f("len_p25_cm"),
            "p50": _f("len_p50_cm"),
            "p75": _f("len_p75_cm"),
            "p90": _f("len_p90_cm"),
            "max": _f("len_max_cm"),
            "mean": _f("len_mean_cm"),
            "n": int(row.get("n_length") or 0),
        },
        "k_factor_distribution": k_factor_block,
        "catch_per_effort": {
            "status": "catch_count_only",
            "catches": sample_size,
            "distinct_anglers": int(row.get("distinct_anglers") or 0),
            "note": (
                "Per-effort normalization (catches / angler-hours) needs outing "
                "duration; reported here as raw catch count + distinct anglers. "
                "True CPUE lands when outing-effort joins are wired into gold."
            ),
        },
        "narrative_hint": (
            f"{sample_size} verified '{species}' catch(es) at "
            f"{row.get('resolved_water_body') or water_body} in the last {window}; "
            f"median length {_f('len_p50_cm')} cm "
            f"(p10 {_f('len_p10_cm')}–p90 {_f('len_p90_cm')}). "
            f"Confidence: {confidence} (n={sample_size}). "
            + ("K-factor pending weight model." if n_kfactor == 0 else "")
        ),
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
    {
        "name": "derbyfish_fingerprint_lookup",
        "description": (
            "Shazam for individual fish. Give it a fish photo (image_url or "
            "image_bytes_b64); it runs the DerbyFish vision worker "
            "(segment -> SigLIP embed) then a k-NN over the verified BHRV fish "
            "fingerprint index and returns the nearest fingerprint match with "
            "confidence (cosine), species, and that fish's catch history "
            "[{angler, date, length_cm, water_body}]. Returns {novel_fish: true} "
            "when nothing clears the confidence threshold. Requires loopback "
            "access to the vision worker (VISION_WORKER_URL) and the analytics "
            "warehouse. Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "Publicly fetchable URL of the fish photo.",
                },
                "image_bytes_b64": {
                    "type": "string",
                    "description": "Base64-encoded image bytes (alternative to image_url).",
                },
                "threshold": {
                    "type": "number",
                    "description": "Cosine match threshold (default 0.90). Below it -> novel_fish.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Neighbors to consider for catch history (1-25, default 5).",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "derbyfish_population_health",
        "description": (
            "Population-health snapshot for a water body x species over a time "
            "window. Answers 'is Lake Champlain healthy this year?'. Returns "
            "length percentiles (cm), catch counts, distinct anglers, K-factor "
            "distribution (marked pending_weight_data until the 3D weight model "
            "ships), raw catch-per-effort, and a sample-size confidence band — "
            "structured for an LLM to narrate. water_body accepts a bow_id uuid "
            "or a water-body name (fuzzy); species is a name (fuzzy). Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "water_body": {
                    "type": "string",
                    "description": "bow_id (uuid) or water-body name, e.g. 'Lake Champlain'.",
                },
                "species": {
                    "type": "string",
                    "description": "Species name, e.g. 'walleye' or 'largemouth bass'.",
                },
                "window": {
                    "type": "string",
                    "description": (
                        "Postgres interval for the lookback, e.g. '365 days', "
                        "'90 days', '1 year'. Default '365 days'."
                    ),
                },
            },
            "required": ["water_body", "species"],
            "additionalProperties": False,
        },
    },
]

TOOL_HANDLERS = {
    "bucket_research": tool_bucket_research,
    "bucket_cite": tool_bucket_cite,
    "bucket_canon_list": tool_bucket_canon_list,
    "derbyfish_fingerprint_lookup": tool_derbyfish_fingerprint_lookup,
    "derbyfish_population_health": tool_derbyfish_population_health,
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
