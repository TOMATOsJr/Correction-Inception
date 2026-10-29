"""Interactive agent protocol: shared claim ledger (blackboard), agent-requested tools, deterministic transitions.

Design rules
  1. Agents PROPOSE actions; the orchestrator VALIDATES and APPLIES them. A model can never mutate state
     directly: it cannot withdraw an opponent's claim, name a span that is not in the transcript, or
     spend more tool calls than its budget. Every rejected action is recorded, not silently dropped.
  2. The ledger is the blackboard. Round 1 is filed blind; every later round's prompt is rendered from the
     ledger, so each agent sees the other's claims and ALL tool evidence before acting again.
  3. Evidence arrives one round later: tools requested in round N are executed by a separate GPU phase and
     are visible from round N+1. This keeps one model in memory at a time (7B agents, then 3B reader).
  4. The arbiter (core.arbitrate) stays pure rules and only ever sees final claims plus their evidence.

Round protocol
  claims  (round 1)  both agents file claims + tool requests, blind to each other
  tools   (any round) executes the requested tools (crop_reread / consistency / span_check / rubric)
  cross   (round 2)  both agents see the full ledger: new / withdraw / revise / counter / defend
  rebut   (round 3)  only for spans where the two agents still propose different values: withdraw / concede / stand
  decide             arbiter over final claims -> variants, ledger, precision
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable

from core import Claim, changed_number, consistency_check, retrieve_rubric, span_check

# actions each round accepts
ROUND_ACTIONS = {1: ("new",), 2: ("new", "withdraw", "revise", "counter", "defend"), 3: ("withdraw", "concede", "stand")}
LIVE = ("open", "final")

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


# ------------------------------------------------------------------ claims
@dataclass
class ClaimRecord:
    """One claim in the ledger. `history` keeps every earlier version so the trace stays auditable."""
    cid: str
    agent: str
    rubric_code: str = ""
    span: str = ""
    proposed: str = ""
    region_type: str = "text"
    rationale: str = ""
    status: str = "open"          # open | withdrawn | conceded | superseded | final
    round_filed: int = 1
    supersedes: str | None = None
    superseded_by: str | None = None
    counters: str | None = None   # cid of the opponent's claim this one disputes
    evidence: dict = field(default_factory=dict)   # latest result per tool
    tool_calls: list = field(default_factory=list)  # tool-log ids
    history: list = field(default_factory=list)

    def live(self) -> bool:
        return self.status in LIVE

    def claims_change(self):
        return changed_number(self.span, self.proposed)

    def to_claim(self) -> Claim:
        """Convert to the arbiter's Claim, with the evidence the arbiter expects."""
        ev = dict(self.evidence)
        rr = ev.get("crop_reread") or {}
        cons = ev.get("consistency") or {}
        return Claim(agent=self.agent, rubric_code=self.rubric_code, span=self.span, proposed=self.proposed,
                     region_type=self.region_type, rationale=self.rationale,
                     evidence={"rid": self.cid,
                               "reread": rr.get("resolved") or rr.get("reading", ""),
                               "reread_conf": float(rr.get("conf") or 0.0),
                               "support": int(cons.get("support") or 0),
                               "n_constraints": int(cons.get("n_constraints") or 0),
                               "reuse_signal": int(cons.get("reuse_signal") or 0)})


# ------------------------------------------------------------------ tools (sandboxed)
class ToolBox:
    """The tools an agent may request. Executed here, never by the model.
    `reread(span, proposed) -> (reading, conf)` wraps the fine-tuned faithful reader on the page region
    with the disputed number masked; None means no reader is available in this phase."""
    NAMES = ("crop_reread", "consistency", "span_check", "rubric")

    def __init__(self, transcript: str, rubric: dict, reread: Callable | None = None, budget: int = 4):
        self.transcript, self.rubric = transcript, rubric
        self.reread, self.budget = reread, budget
        self.used = Counter()   # agent -> calls used

    def remaining(self, agent: str) -> int:
        return max(0, self.budget - self.used[agent])

    def call(self, agent: str, name: str, args: dict) -> dict:
        """Returns {"tool", "args", "ok", "result"/"error"}. Never raises."""
        args = args if isinstance(args, dict) else {}
        args = {k: (v if isinstance(v, (int, float, bool)) or v is None else str(v)) for k, v in args.items()}
        if name not in self.NAMES:
            return {"tool": name, "args": args, "ok": False, "error": f"unknown tool (available: {', '.join(self.NAMES)})"}
        if self.used[agent] >= self.budget:
            return {"tool": name, "args": args, "ok": False, "error": f"tool budget exhausted ({self.budget} per agent per exam)"}
        self.used[agent] += 1
        try:
            if name == "span_check":
                return {"tool": name, "args": args, "ok": True, "result": {"present": span_check(self.transcript, args.get("span", ""))}}
            if name == "consistency":
                span, proposed = args.get("span", ""), args.get("proposed", "")
                if not span_check(self.transcript, span):
                    return {"tool": name, "args": args, "ok": False, "error": "span not found in transcript"}
                return {"tool": name, "args": args, "ok": True, "result": consistency_check(self.transcript, span, proposed)}
            if name == "rubric":
                crit = retrieve_rubric(self.rubric, args.get("code", ""))
                return {"tool": name, "args": args, "ok": crit is not None,
                        "result": crit or {}, **({} if crit else {"error": f"no rubric code {args.get('code')!r}"})}
            if name == "crop_reread":
                span, proposed = args.get("span", ""), args.get("proposed", "")
                if self.reread is None:
                    return {"tool": name, "args": args, "ok": False, "error": "reader not available in this phase"}
                if not span_check(self.transcript, span):
                    return {"tool": name, "args": args, "ok": False, "error": "span not found in transcript"}
                if changed_number(span, proposed) is None:
                    return {"tool": name, "args": args, "ok": False,
                            "error": "crop_reread needs a claim that changes exactly one number"}
                reading, conf = self.reread(span, proposed)
                return {"tool": name, "args": args, "ok": True,
                        "result": {"reading": reading, "conf": float(conf),
                                   "resolved": (proposed if _norm(reading) == _norm(proposed) else
                                                span if _norm(reading) == _norm(span) else reading)}}
        except Exception as e:  # a tool fault must never kill the run
            return {"tool": name, "args": args, "ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        return {"tool": name, "args": args, "ok": False, "error": "unreachable"}


# ------------------------------------------------------------------ ledger
@dataclass
class Ledger:
    exam: str
    transcript: str
    claims: dict[str, ClaimRecord] = field(default_factory=dict)
    tool_log: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    _next: int = 1

    def add(self, agent: str, d: dict, round_no: int, **extra) -> ClaimRecord:
        cid = f"#{self._next}"
        self._next += 1
        rec = ClaimRecord(cid=cid, agent=agent, rubric_code=str(d.get("rubric_code", "")), span=str(d.get("span", "")),
                          proposed=str(d.get("proposed", "")), region_type=str(d.get("region_type", "text")),
                          rationale=str(d.get("rationale", d.get("reason", ""))), round_filed=round_no, **extra)
        rec.history.append({"round": round_no, "span": rec.span, "proposed": rec.proposed, "rationale": rec.rationale})
        self.claims[cid] = rec
        return rec

    def live(self) -> list[ClaimRecord]:
        return [c for c in self.claims.values() if c.live()]

    def final_claims(self) -> list[Claim]:
        return [c.to_claim() for c in self.live()]

    def of(self, agent: str) -> list[ClaimRecord]:
        return [c for c in self.claims.values() if c.agent == agent]

    def attach_tool(self, rec: ClaimRecord, res: dict, rid: str):
        rec.tool_calls.append(rid)
        self.tool_log.append(res)
        if res.get("ok"):
            rec.evidence[res["tool"]] = res["result"]
        else:
            rec.evidence.setdefault(f"{res['tool']}_error", res.get("error", ""))

    # -- rendering for prompts
    def render(self, only: list[str] | None = None, max_evidence: int = 3) -> str:
        rows = [c for c in self.claims.values() if (only is None or c.cid in only) and c.status != "superseded"]
        if not rows:
            return "(empty)"
        out = []
        for c in rows:
            head = (f"{c.cid} [{c.agent}, filed r{c.round_filed}, {c.status}] code={c.rubric_code or '?'}  "
                    f"span: `{c.span}`  ->  proposed: `{c.proposed}`")
            if c.counters:
                head += f"  (disputes {c.counters})"
            if c.supersedes:
                head += f"  (replaces {c.supersedes})"
            out.append(head)
            if c.rationale:
                out.append(f"    rationale: {c.rationale[:300]}")
            ev = []
            for name, val in list(c.evidence.items())[:max_evidence]:
                ev.append(f"{name}=" + (json.dumps(val, ensure_ascii=False)[:180] if isinstance(val, dict) else str(val)[:180]))
            if ev:
                out.append("    evidence: " + "; ".join(ev))
        return "\n".join(out)


# ------------------------------------------------------------------ state transitions
def apply_new(led: Ledger, agent: str, d: dict, round_no: int, counters: str | None = None) -> tuple[ClaimRecord | None, str | None]:
    span, proposed = str(d.get("span", "")), str(d.get("proposed", ""))
    if not span or not proposed:
        return None, "claim needs both span and proposed"
    if not span_check(led.transcript, span):
        return None, f"span not found in the transcript: {span[:60]!r}"
    if _norm(span) == _norm(proposed):
        return None, "span and proposed are identical"
    return led.add(agent, d, round_no, counters=counters), None


def apply_actions(led: Ledger, agent: str, round_no: int, actions: list) -> list[dict]:
    """Validate and apply one agent's action list. Returns the per-action outcome (applied / rejected + why).
    Nothing here trusts the model: ownership, span existence and round legality are all enforced."""
    allowed = ROUND_ACTIONS[round_no]
    results = []
    for a in actions if isinstance(actions, list) else []:
        if not isinstance(a, dict):
            results.append({"action": str(a)[:40], "ok": False, "error": "action must be an object"})
            led.rejected.append(results[-1])
            continue
        name = str(a.get("action", "")).lower()
        cid = str(a.get("cid", "")).strip()
        if name not in allowed:
            results.append({"action": name, "cid": cid, "ok": False,
                            "error": f"action {name!r} not allowed in round {round_no} (allowed: {', '.join(allowed)})"})
            led.rejected.append(results[-1])
            continue
        rec = led.claims.get(cid)
        if name != "new" and rec is None:
            results.append({"action": name, "cid": cid, "ok": False, "error": "unknown claim id"})
            led.rejected.append(results[-1])
            continue

        if name == "new":
            new, err = apply_new(led, agent, a, round_no)
            results.append({"action": name, "cid": new.cid if new else cid, "ok": new is not None, **({"error": err} if err else {})})
        elif name == "withdraw":
            if rec.agent != agent:
                results.append({"action": name, "cid": cid, "ok": False, "error": "you can only withdraw your own claim"})
            elif not rec.live():
                results.append({"action": name, "cid": cid, "ok": False, "error": f"claim is already {rec.status}"})
            else:
                rec.status, rec.rationale = "withdrawn", str(a.get("reason", rec.rationale))
                rec.history.append({"round": round_no, "action": "withdraw", "reason": str(a.get("reason", ""))})
                results.append({"action": name, "cid": cid, "ok": True})
        elif name == "concede":
            if rec.agent != agent:
                results.append({"action": name, "cid": cid, "ok": False, "error": "you can only concede your own claim"})
            elif not rec.live():
                results.append({"action": name, "cid": cid, "ok": False, "error": f"claim is already {rec.status}"})
            else:
                rec.status = "conceded"
                rec.history.append({"round": round_no, "action": "concede", "reason": str(a.get("reason", ""))})
                results.append({"action": name, "cid": cid, "ok": True})
        elif name == "stand":
            if rec.agent != agent:
                results.append({"action": name, "cid": cid, "ok": False, "error": "you can only defend your own claim"})
            elif not rec.live():
                results.append({"action": name, "cid": cid, "ok": False, "error": f"claim is already {rec.status}"})
            else:
                rec.status = "final"
                rec.rationale = (rec.rationale + " | round 3: " + str(a.get("reason", "")))[:600]
                rec.history.append({"round": round_no, "action": "stand", "reason": str(a.get("reason", ""))})
                results.append({"action": name, "cid": cid, "ok": True})
        elif name == "defend":
            if rec.agent != agent:
                results.append({"action": name, "cid": cid, "ok": False, "error": "you can only defend your own claim"})
            elif not rec.live():
                results.append({"action": name, "cid": cid, "ok": False, "error": f"claim is already {rec.status}"})
            else:
                rec.rationale = (rec.rationale + " | " + str(a.get("reason", "")))[:600]
                rec.history.append({"round": round_no, "action": "defend", "reason": str(a.get("reason", ""))})
                results.append({"action": name, "cid": cid, "ok": True})
        elif name == "counter":
            if rec.agent == agent:
                results.append({"action": name, "cid": cid, "ok": False, "error": "counter is for the OTHER agent's claim; use revise for your own"})
            elif not rec.live():
                results.append({"action": name, "cid": cid, "ok": False, "error": f"that claim is already {rec.status}"})
            else:
                d = {**a, "span": a.get("span") or rec.span}  # a counter disputes the same span by default
                new, err = apply_new(led, agent, d, round_no, counters=cid)
                results.append({"action": name, "cid": new.cid if new else cid, "ok": new is not None, **({"error": err} if err else {})})
        elif name == "revise":
            if rec.agent != agent:
                results.append({"action": name, "cid": cid, "ok": False, "error": "you can only revise your own claim"})
            elif not rec.live():
                results.append({"action": name, "cid": cid, "ok": False, "error": f"claim is already {rec.status}"})
            else:
                d = {**rec.__dict__, **{k: v for k, v in a.items() if k in ("span", "proposed", "rubric_code", "region_type")}}
                new, err = apply_new(led, agent, d, round_no, counters=rec.counters)
                if new:
                    rec.status, rec.superseded_by = "superseded", new.cid
                    new.supersedes = cid
                results.append({"action": name, "cid": new.cid if new else cid, "ok": new is not None, **({"error": err} if err else {})})
        if not results[-1]["ok"]:
            led.rejected.append(results[-1])
    return results


# ------------------------------------------------------------------ replay -> ledger
LEDGERS = ("r1", "tools", "r2", "r3")   # the JSONL files that make up the shared ledger


def build_ledger(exam: str, transcript: str, claims_recs: list, tool_recs: list, action_recs: list) -> Ledger:
    """Replay the JSONL phases into a ledger. Deterministic: same log -> same ledger -> same cids.
    Claim/action records are replayed in (round, advocate, auditor) order, which is the order the
    orchestrator wrote them, so the cids assigned here match the ones the tool log refers to."""
    led = Ledger(exam=exam, transcript=transcript)
    rank = {"advocate": 0, "auditor": 1}
    recs = [{**r, "round": int(r.get("round", 1))} for r in list(claims_recs) + list(action_recs)]
    for rec in sorted(recs, key=lambda r: (r["round"], rank.get(r.get("agent", ""), 9))):
        agent, round_no = rec.get("agent", ""), rec["round"]
        for d in rec.get("claims", []):
            apply_new(led, agent, d, round_no)
        for a in rec.get("actions", []):
            apply_actions(led, agent, round_no, [a])
    for t in tool_recs:  # tool results land on the claim they were requested for
        rec = led.claims.get(t.get("cid", ""))
        if rec is not None:
            led.attach_tool(rec, t, t.get("id", ""))
    return led


def conflict_groups(led: Ledger) -> list[list[str]]:
    """Live claims that dispute the same value with different proposals (overlapping spans count:
    '8 * 4' and '8 * 4 = 12' conflict when they propose different replacements)."""
    rows = led.live()
    seen, groups = set(), []
    for i, a in enumerate(rows):
        group = {a.cid}
        for j, b in enumerate(rows):
            if i == j:
                continue
            sa, sb = _norm(a.span), _norm(b.span)
            if (sa and sb and (sa in sb or sb in sa)) and _norm(a.proposed) != _norm(b.proposed):
                group.add(b.cid)
        key = frozenset(group)
        if len(group) > 1 and key not in seen:
            seen.add(key)
            groups.append(sorted(group))
    return groups


# ------------------------------------------------------------------ prompt parsing
# Models often emit LaTeX with single backslashes (\frac, \Omega) inside JSON strings.
# \O is an invalid JSON escape (parse error), but \f, \t, \n, \b, \r are VALID escapes
# that silently mangle the text (\frac -> formfeed + "rac", \text -> tab + "ext").
# Repair: re-escape any single backslash that starts a LaTeX-like sequence, then retry.
_SUSPECT = re.compile(r'\\[ftnbr](?=[a-zA-Z])')   # raw \frac / \text / \nu / \beta / \rho


def _repair(text: str) -> str:
    text = re.sub(r'(?<!\\)\\u(?![0-9a-fA-F]{4})', r'\\\\u', text)   # \mu, \nu (invalid \u)
    text = re.sub(r'(?<!\\)\\(?=[a-tv-zA-Z])', r'\\\\', text)        # \frac, \text, \rho, ... ('u' kept for \uXXXX)
    text = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', text)       # \O, \G, \{, ...
    return text


def _loads(text: str):
    try:
        data = json.loads(text)
        if not _SUSPECT.search(text):   # parsed clean -> trust it
            return data
    except Exception:
        pass
    return json.loads(_repair(text))


def _salvage_objects(text: str) -> list[dict]:
    """Last resort for structurally broken / truncated JSON (responses are stored
    truncated, objects may be closed early): brace-match every balanced {...} at any
    depth and parse each on its own, so one malformed element does not discard the
    whole response."""
    out, stack, in_str, esc = [], [], False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            stack.append(i)
        elif ch == '}' and stack:
            s = stack.pop()
            try:
                obj = _loads(text[s:i + 1])
            except Exception:
                continue
            if isinstance(obj, dict) and obj not in out:
                out.append(obj)
    return out


def _is_action(a: dict) -> bool:
    if a.get("action"):
        return True
    # a filed claim (vs. a tool-call args fragment) carries claim metadata too
    return "span" in a and "proposed" in a and bool({"rubric_code", "region_type", "rationale"} & set(a))


def parse_actions(text: str) -> list[dict]:
    """Parse an agent's JSON. Accepts {"actions": [...]}, {"claims": [...]} or a bare list."""
    if not text or text.startswith("Error:"):
        return []
    for cand in (text, *re.findall(r"```(?:json)?(.*?)```", text, re.S), *re.findall(r"[\{\[].*[\}\]]", text, re.S)):
        try:
            data = _loads(cand.strip())
        except Exception:
            continue
        if isinstance(data, dict):
            for k in ("actions", "claims"):
                if isinstance(data.get(k), list):
                    # some models nest {"tool_requests": [...]} inside the action list
                    return [a for a in data[k] if isinstance(a, dict) and "tool_requests" not in a]
        if isinstance(data, list):
            return [a for a in data if isinstance(a, dict) and "tool_requests" not in a]
    # structurally broken JSON: keep whatever well-formed action objects can be matched out
    return [a for a in _salvage_objects(text) if _is_action(a)]


def parse_tool_requests(text: str) -> list[dict]:
    if not text or text.startswith("Error:"):
        return []
    for cand in (text, *re.findall(r"```(?:json)?(.*?)```", text, re.S), *re.findall(r"\{.*\}", text, re.S)):
        try:
            data = _loads(cand.strip())
        except Exception:
            continue
        if isinstance(data, dict):
            if isinstance(data.get("tool_requests"), list):
                return [t for t in data["tool_requests"] if isinstance(t, dict)]
            for a in data.get("actions", []) if isinstance(data.get("actions"), list) else []:
                if isinstance(a, dict) and isinstance(a.get("tool_requests"), list):
                    return [t for t in a["tool_requests"] if isinstance(t, dict)]
    # structurally broken JSON: keep tool requests matched out of intact fragments
    reqs = []
    for obj in _salvage_objects(text):
        if isinstance(obj.get("tool_requests"), list):
            reqs += [t for t in obj["tool_requests"] if isinstance(t, dict)]
    return reqs


def parse_single_action(text: str) -> dict | None:
    """Round 3 answers with one action object: {"action": "stand"|"concede"|"withdraw", ...}."""
    if not text or text.startswith("Error:"):
        return None
    for cand in (text, *re.findall(r"```(?:json)?(.*?)```", text, re.S), *re.findall(r"\{.*\}", text, re.S)):
        try:
            data = _loads(cand.strip())
            if isinstance(data, dict) and str(data.get("action", "")).lower() in ("stand", "concede", "withdraw"):
                return data
        except Exception:
            continue
    for obj in _salvage_objects(text):   # broken JSON: keep a well-formed rebuttal if there is one
        if str(obj.get("action", "")).lower() in ("stand", "concede", "withdraw"):
            return obj
    return None


def ledger_json(led: Ledger) -> str:
    return json.dumps({"exam": led.exam, "claims": [asdict(c) for c in led.claims.values()],
                       "tool_log": led.tool_log, "rejected": led.rejected}, indent=2, ensure_ascii=False)
