#!/usr/bin/env python
"""Synthesise a multi-tool-calling SFT corpus for Qwen3.5-4B.

Why this exists
---------------
Measured 2026-08-21 with local-model-eval `multi_tool_use`: Qwen3.5-4B scored
5/14, and the per-turn call distribution was {0 calls: 3, 1 call: 11}. It NEVER
emitted two calls in one response. Every single call it did make named the
right tool with the right arguments — so tool selection and argument
extraction are already fine. Exactly one behaviour is missing: emitting
several independent calls in a single assistant turn.

That makes this a narrow adapter, not a general capability push.

Two rules shape the mix
-----------------------
1. NO LEAKAGE. The eval uses Kaleb / Emmitt / Fort Worth / Dallas / Austin.
   None of those appear here — different people, different cities, different
   phrasings. Training on the test is how you get an adapter that scores well
   and helps nothing.

2. DON'T BREAK RESTRAINT. The model currently passes the "answer without
   calling anything" cases, which is the behaviour that stops an assistant
   messaging your family unprompted. Training only on multi-call examples
   would teach it to call tools constantly, trading one failure for a worse
   one. So the corpus is deliberately mixed:

       ~45% multi-call   (the target behaviour)
       ~30% single-call  (preserve correct restraint on simple asks)
       ~25% no-call      (preserve refusal to act when nothing was asked)

    python gen_multitool_sft.py --out data/multitool_sft.jsonl --n 600
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --- vocabulary, disjoint from the eval set --------------------------------
PEOPLE = ["Marcus", "Priya", "Devon", "Yuki", "Rosa", "Nate", "Imani", "Tobias",
          "Lena", "Omar", "Sasha", "Bea", "Cyrus", "Nadia"]
CITIES = ["Tulsa", "Boise", "Akron", "Fresno", "Mobile", "Spokane", "Duluth",
          "Reno", "Tacoma", "Wichita", "Savannah", "Bozeman"]
TIMES = ["7am", "8:15am", "noon", "2pm", "4:45pm", "6pm", "8:30pm", "10pm"]
ERRANDS = ["take out the recycling", "start homework", "feed the dog",
           "pack a lunch", "call the dentist", "water the plants",
           "charge the tablet", "put laundry away", "practice piano"]
NOTES = ["dinner is ready", "the package arrived", "practice is cancelled",
         "we're leaving in ten minutes", "good luck today", "call me back"]
FILES = ["notes/roadmap.md", "src/server.ts", "data/inventory.csv",
         "docs/setup.md", "config/settings.toml"]
SHELL = [("count the lines in every markdown file", "*.md"),
         ("show disk usage for the current folder", "du"),
         ("list files changed in the last day", "find"),
         ("show the running python processes", "python")]

TOOLS = [
    {"type": "function", "function": {"name": "execute_shell",
        "description": "Run a shell command and return its stdout.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file",
        "description": "Read and return a file's contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file",
        "description": "Write text content to a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                       "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "set_reminder",
        "description": "Schedule a reminder for a person at a time.",
        "parameters": {"type": "object", "properties": {"person": {"type": "string"},
                       "time": {"type": "string"}, "message": {"type": "string"}},
                       "required": ["person", "time", "message"]}}},
    {"type": "function", "function": {"name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                       "unit": {"type": "string", "enum": ["f", "c"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "send_message",
        "description": "Send a message to a person on their device.",
        "parameters": {"type": "object", "properties": {"to": {"type": "string"},
                       "body": {"type": "string"}}, "required": ["to", "body"]}}},
    {"type": "function", "function": {"name": "list_calendar",
        "description": "List calendar events for a person on a given day.",
        "parameters": {"type": "object", "properties": {"person": {"type": "string"},
                       "day": {"type": "string"}}, "required": ["person"]}}},
]

# Restraint examples must be as VARIED as the action examples, or the class
# starves under prompt dedup and the adapter learns to call tools constantly.
# Three shapes, each templated: plain knowledge, explicit do-not-act, and
# "describe the action instead of taking it".
FACTS = [
    ("What's the capital of {}?", {"Portugal": "Lisbon.", "Peru": "Lima.",
     "Norway": "Oslo.", "Kenya": "Nairobi.", "Vietnam": "Hanoi.",
     "Uruguay": "Montevideo.", "Morocco": "Rabat.", "Nepal": "Kathmandu."}),
    ("How many {} are in a {}?", {"days|leap year": "366.", "hours|day": "24.",
     "minutes|hour": "60.", "sides|hexagon": "Six.", "cards|standard deck": "52.",
     "innings|baseball game": "Nine, normally.", "strings|guitar": "Six, usually.",
     "players|soccer team": "Eleven on the field.", "bones|adult human body": "206.",
     "continents|world": "Seven.", "weeks|year": "52.", "letters|alphabet": "26."}),
    ("Who wrote {}?", {"Moby-Dick": "Herman Melville.", "Frankenstein": "Mary Shelley.",
     "The Odyssey": "Homer, traditionally.", "Dracula": "Bram Stoker.",
     "Beloved": "Toni Morrison.", "Dune": "Frank Herbert.",
     "The Hobbit": "J.R.R. Tolkien.", "Kindred": "Octavia Butler."}),
]
CONCEPTS = [
    ("the difference between a reminder and a calendar event",
     "A reminder nudges one person at a time; a calendar event blocks a span and others can see it."),
    ("what a tool call is", "The model asking the host to run a named function with structured arguments."),
    ("why an assistant should confirm before sending a message",
     "Because a sent message can't be unsent, and a wrong one costs trust."),
    ("what makes a good reminder message",
     "Short, specific, and phrased as the action to take."),
    ("when a shell command is the wrong tool",
     "When the thing you want is a decision, not an output."),
    ("why parallel tool calls matter",
     "One turn does the whole request instead of making the user ask twice."),
    ("the risk of an assistant that acts without being asked",
     "It turns a helpful system into one you have to supervise."),
    ("why a reminder needs a time", "Without one it's a note, not a nudge."),
    ("what makes a tool call fail silently",
     "Emitting it as prose — the caller sees no call at all."),
    ("when to ask instead of act", "When the action is hard to undo."),
    ("why an assistant should say what it did",
     "So a wrong action is caught in seconds instead of days."),
    ("the difference between a notification and an interruption",
     "One waits to be read; the other decides it can't."),
    ("why fewer tools is often better", "Every extra tool is another way to be wrong."),
    ("what a good default looks like", "The choice you'd make anyway, made without asking."),
]
DONT_ACT = [
    ("Don't do anything yet — just tell me what you'd need to {}.", {
        "set a reminder": "A person, a time, and the message.",
        "send a message": "Who it goes to, and what it says.",
        "check the weather": "A city, and whether you want F or C.",
        "look up a calendar": "Whose calendar, and which day."}),
    ("If I asked you to {}, what would you do first? Don't actually do it.", {
        "clean up my downloads folder": "List what's there before deleting anything.",
        "message the whole family": "Confirm the recipient list with you first.",
        "rewrite a config file": "Read the current one and show you the diff.",
        "cancel an event": "Check the calendar to make sure I have the right one."}),
    ("Hypothetically, how would you handle {}? Don't take any action.", {
        "two reminders at the same time": "Set both; they don't conflict.",
        "a message to someone not in the household": "Ask you to confirm the recipient.",
        "a weather request with no city": "Ask which city before calling anything.",
        "a request to delete files": "Show the list and wait for a yes.",
        "an ambiguous name": "Ask which person you meant.",
        "a reminder in the past": "Flag it and ask for the intended time."}),
]


def _call(name: str, **args) -> dict:
    """Arguments stay a DICT, not a JSON string.

    The OpenAI wire format returns `arguments` as a serialised string, so it is
    tempting to store one here — but the Qwen chat template iterates it with
    `.items()` at render time and dies on a string. Training data is rendered
    by the template, not by the API.
    """
    return {"type": "function", "function": {"name": name, "arguments": args}}


def _reminder(rng):
    p, t, e = rng.choice(PEOPLE), rng.choice(TIMES), rng.choice(ERRANDS)
    return f"remind {p} to {e} at {t}", _call("set_reminder", person=p, time=t, message=e.capitalize())


def _weather(rng):
    c = rng.choice(CITIES)
    unit = rng.choice(["f", "c"])
    return f"tell me the weather in {c}", _call("get_weather", city=c, unit=unit)


def _message(rng):
    p, n = rng.choice(PEOPLE), rng.choice(NOTES)
    return f"message {p} that {n}", _call("send_message", to=p, body=n.capitalize())


def _calendar(rng):
    p = rng.choice(PEOPLE)
    day = rng.choice(["today", "tomorrow", "Friday"])
    return f"show {p}'s calendar for {day}", _call("list_calendar", person=p, day=day)


def _readfile(rng):
    f = rng.choice(FILES)
    return f"read {f}", _call("read_file", path=f)


def _shell(rng):
    desc, frag = rng.choice(SHELL)
    return desc, _call("execute_shell", command=frag)


MAKERS = [_reminder, _weather, _message, _calendar, _readfile, _shell]

JOINERS = [" and ", ", and also ", ". Then ", ", plus ", ". Also "]


OPINIONS = [
    ("Is it better to {} or {}?", [
        ("remind someone early", "remind them exactly on time",
         "Early, if the task needs setup; on time otherwise."),
        ("send one long message", "send several short ones",
         "One clear message. Several reads as nagging."),
        ("ask before every action", "ask only before irreversible ones",
         "Only before irreversible ones — the rest is friction."),
        ("show all the detail", "show the summary first",
         "Summary first, detail on request."),
        ("act fast and correct later", "verify then act",
         "Verify, when the action touches other people."),
        ("keep every memory", "keep only what stays true",
         "Only what stays true. Stale facts are worse than none."),
    ]),
]


def _no_call(rng) -> tuple[str, str]:
    shape = rng.choice(("fact", "concept", "dont_act", "opinion"))
    if shape == "opinion":
        tmpl, rows = rng.choice(OPINIONS)
        a, b, ans = rng.choice(rows)
        lead = rng.choice(["", "Quick one: ", "No action needed — ", "Just curious: "])
        return lead + tmpl.format(a, b), ans
    if shape == "fact":
        tmpl, table = rng.choice(FACTS)
        key = rng.choice(list(table))
        return tmpl.format(*key.split("|")), table[key]
    if shape == "concept":
        topic, ans = rng.choice(CONCEPTS)
        lead = rng.choice(["Explain {} in one sentence.", "In one line, what is {}?",
                           "Briefly: {}.", "Tell me {} — no actions.",
                           "In your own words, {}?", "One sentence on {}, please.",
                           "Without doing anything, explain {}.",
                           "Help me understand {}."])
        return lead.format(topic), ans
    tmpl, table = rng.choice(DONT_ACT)
    key = rng.choice(list(table))
    return tmpl.format(key), table[key]


def build(rng, kind: str) -> dict:
    """Build one example of an explicitly requested class.

    The class is a PARAMETER, not a dice roll: rolling meant the restraint
    class starved against prompt-dedup (8 of 600 on the first pass) and the
    adapter would have learned to over-call.
    """
    if kind == "multi":
        k = 2 if rng.random() < 0.75 else 3
        makers = rng.sample(MAKERS, k) if rng.random() < 0.8 else [rng.choice(MAKERS)] * k
        parts, calls = [], []
        for m in makers:
            text, call = m(rng)
            parts.append(text)
            calls.append(call)
        prompt = rng.choice(JOINERS).join(parts)
        prompt = prompt[0].upper() + prompt[1:] + "."
        return {"messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": "", "tool_calls": calls}],
                "tools": TOOLS, "kind": f"multi{k}"}
    if kind == "single":
        text, call = rng.choice(MAKERS)(rng)
        prompt = text[0].upper() + text[1:] + "."
        return {"messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": "", "tool_calls": [call]}],
                "tools": TOOLS, "kind": "single"}
    prompt, answer = _no_call(rng)
    return {"messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": answer}],
            "tools": TOOLS, "kind": "none"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/multitool_sft.jsonl"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260821)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # Explicit quotas. Restraint is 25% by construction, not by luck.
    quotas = {"multi": round(args.n * 0.45), "single": round(args.n * 0.30)}
    quotas["none"] = args.n - sum(quotas.values())

    rows, seen = [], set()
    for kind, quota in quotas.items():
        made, attempts = 0, 0
        while made < quota and attempts < quota * 60:
            attempts += 1
            row = build(rng, kind)
            key = row["messages"][0]["content"]
            if key in seen:                # exact-duplicate prompts teach nothing
                continue
            seen.add(key)
            rows.append(row)
            made += 1
        if made < quota:
            raise SystemExit(f"{kind}: only {made}/{quota} unique prompts available — "
                             f"add more templates before training")
    rng.shuffle(rows)

    # Leakage guard: the eval's vocabulary must not appear anywhere in here.
    banned = ["Kaleb", "Emmitt", "Fort Worth", "Dallas", "Austin",
              "pipeline_config.yaml", "accountant"]
    blob = json.dumps(rows)
    hits = [b for b in banned if b.lower() in blob.lower()]
    if hits:
        raise SystemExit(f"LEAKAGE: eval vocabulary present in training data: {hits}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8")

    mix: dict[str, int] = {}
    for r in rows:
        mix[r["kind"]] = mix.get(r["kind"], 0) + 1
    print(f"wrote {len(rows)} examples -> {args.out}")
    print(f"  mix: {dict(sorted(mix.items()))}")
    print("  leakage check: clean (no eval vocabulary present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
