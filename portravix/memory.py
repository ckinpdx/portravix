"""Long-term memory across sessions, kept in one Markdown file that people (or another agent) can read and edit.

The file has three standard sections, each a list of "- " lines:
    ## About <user>                        facts about the user
    ## Things you've said about yourself   what the avatar has claimed about itself, so it stays consistent
    ## Recent conversations                one dated summary line per conversation
Other sections someone adds are kept, and given to the LLM like the facts.

Each finished exchange goes into a private pending file (cache/). A curator pass sends the pending exchanges to the same
LLM when the conversation has been quiet for a while, when the launcher stops the avatar, or at the next start for
anything left over. The LLM answers in a fixed JSON structure (summary, new facts, corrections, self-claims) and the code
applies it line by line: the model never rewrites the file, so it cannot drop what is already there.
"""
import datetime
import hashlib
import json
import os
import threading
import time
import urllib.request

from . import config as C

LOG_SECTION = "Recent conversations"
SELF_SECTION = "Things you've said about yourself"
HEADER_MARK = "<!-- Portravix"
BATCH_TOKENS = 6000  # exchanges per curator call: the avatar's LLM may have only a 16k context
ITEM_CHARS = 240     # longest line the curator may add

CURATOR_SYSTEM = (
    "You keep the long-term memory of a live talking avatar that chats with {user}. You get what is already "
    "remembered and a new conversation between {user} and the avatar. Answer with JSON only:\n"
    '- "summary": one sentence, under 30 words, on what this conversation was about, as a note for the avatar.\n'
    '- "new_facts": lasting facts about {user} from the new conversation that are not already remembered: people, '
    "pets, work, projects, plans, likes and dislikes, events that matter. One short sentence each, naming {user}. "
    "Leave out small talk, passing moods and anything only true today.\n"
    '- "corrections": remembered lines that the new conversation shows are wrong or out of date. "old" is the '
    'existing line copied exactly; "new" is the corrected line, or "" to delete it.\n'
    '- "self_claims": specific things the avatar said about itself (likes, history, opinions, habits) that are not '
    'already remembered. One short sentence each, starting with "You".\n'
    "Use empty lists when there is nothing. Record only what was actually said.")

_ITEMS = {"type": "array", "maxItems": 12, "items": {"type": "string", "maxLength": ITEM_CHARS}}
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 300},
        "new_facts": _ITEMS,
        "corrections": {"type": "array", "maxItems": 12, "items": {
            "type": "object",
            "properties": {"old": {"type": "string", "maxLength": 400}, "new": {"type": "string", "maxLength": ITEM_CHARS}},
            "required": ["old", "new"]}},
        "self_claims": _ITEMS,
    },
    "required": ["summary", "new_facts", "corrections", "self_claims"],
}


def tokens(text):
    return len(text) / 4  # rough estimate for English, as for the persona


def _norm(line):
    """A line compared loosely: no bullet, case, extra spaces or final full stop."""
    line = line.strip()
    if line.startswith("- "):
        line = line[2:]
    return " ".join(line.lower().split()).rstrip(".")


def _clean(text):
    text = " ".join(str(text).split())
    if text.startswith("- "):
        text = text[2:]
    return text[:ITEM_CHARS].strip()


def part_of_day(dt):
    return ("night", "morning", "afternoon", "evening")[(5 <= dt.hour < 12) + 2 * (12 <= dt.hour < 17) +
                                                         3 * (17 <= dt.hour < 22)]


def ago(seconds):
    minutes = max(seconds, 60) / 60
    if minutes < 90:
        return f"{round(minutes)} minute{'s' * (round(minutes) != 1)} ago"
    hours = minutes / 60
    if hours < 36:
        return f"{round(hours)} hours ago"
    days = hours / 24
    if days < 14:
        return f"{round(days)} days ago"
    if days < 60:
        return f"{round(days / 7)} weeks ago"
    return f"{round(days / 30)} months ago"


def now_text(dt=None):
    """"Saturday 24 October 2026, 7:42 pm"."""
    dt = dt or datetime.datetime.now()
    return f"{dt:%A} {dt.day} {dt:%B %Y}, {dt.hour % 12 or 12}:{dt:%M} {'am' if dt.hour < 12 else 'pm'}"


def _parse(text):
    """(header lines without the generated comment, [[title, lines]])."""
    header, sections, in_mark = [], [], False
    for line in text.splitlines():
        if not sections and (in_mark or line.strip().startswith(HEADER_MARK)):
            in_mark = "-->" not in line
            continue
        if line.startswith("## "):
            sections.append([line[3:].strip(), []])
        elif sections:
            sections[-1][1].append(line)
        else:
            header.append(line)
    return header, sections


def _items(lines):
    return [line for line in lines if line.startswith("- ")]


class Memory:
    def __init__(self, path, url="", model="", user="the user", budget_tokens=1200, summaries=5, keep_summaries=20,
                 gap_minutes=10, state_dir=None, log=print):
        """path: the memory file (created if missing). url/model: the chat server the curator asks.
        budget_tokens: how much of the file (all but the summaries) the LLM is given, newest lines first.
        summaries: conversation summaries given to the LLM; keep_summaries: kept in the file (older: the archive).
        gap_minutes: a quiet gap this long between exchanges starts a new conversation (its own summary)."""
        self.path, self.url, self.model, self.user, self.log = path, url, model, user, log
        self.facts_section = f"About {user}"
        self.budget, self.summaries, self.keep = float(budget_tokens), int(summaries), int(keep_summaries)
        self.gap = float(gap_minutes) * 60
        root, ext = os.path.splitext(path)
        self.archive_path = f"{root}_archive{ext or '.md'}"
        key = hashlib.sha1(os.path.abspath(path).lower().encode()).hexdigest()[:8]
        state_dir = state_dir or os.path.join(C.ROOT, "cache")
        os.makedirs(state_dir, exist_ok=True)
        self.pending_path = os.path.join(state_dir, f"memory_pending_{key}.jsonl")
        self.state_path = os.path.join(state_dir, f"memory_state_{key}.json")
        self._lock = threading.Lock()
        self._mtime, self._block = None, ""
        self.last_turn = self._read_state().get("last_turn")
        self.previous_turn = self.last_turn  # when the conversation before this session ended
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._write([f"# Memory"], [])
            log(f"[memory] started a new memory file: {path}")

    # ----- the file -----

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                return _parse(f.read())
        except FileNotFoundError:  # deleted while running: start again
            return ["# Memory"], []

    def _write(self, header, sections):
        """Write the file in one step, with the standard sections present and a fresh size note."""
        titles = [s[0] for s in sections]
        for title in (self.facts_section, SELF_SECTION, LOG_SECTION):
            if title not in titles:
                sections.append([title, []])
        size = sum(tokens(line) for title, lines in sections if title != LOG_SECTION for line in _items(lines))
        note = (f"{HEADER_MARK} keeps this file: it adds what it learns after each conversation, and anyone may edit it. "
                f'One "- " line per item; keep the ## headings. The avatar is given the newest lines that fit '
                f"~{self.budget:.0f} tokens across every section except \"{LOG_SECTION}\", plus the {self.summaries} "
                f"newest conversation summaries. Size at the last update: ~{size:.0f} of {self.budget:.0f} tokens. -->")
        header = [line for line in header]
        while header and not header[-1].strip():
            header.pop()
        title_at = next((i for i, line in enumerate(header) if line.startswith("# ")), None)
        if title_at is None:
            header.insert(0, "# Memory")
            title_at = 0
        header[title_at + 1:title_at + 1] = ["", note]
        out = header + [""]
        for title, lines in sections:
            while lines and not lines[-1].strip():
                lines = lines[:-1]
            while lines and not lines[0].strip():
                lines = lines[1:]
            out += [f"## {title}", *lines, ""]
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(out))
        for attempt in range(20):  # someone may have the file open for a moment
            try:
                os.replace(tmp, self.path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.1)
        self._mtime = None

    def block(self):
        """What the LLM is given, re-read whenever the file changes: the newest lines that fit the budget, then the
        newest summaries. Empty when there is nothing yet."""
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return ""
        if mtime == self._mtime:
            return self._block
        self._mtime = mtime
        try:
            _, sections = self._read()
        except OSError as e:
            self.log(f"[memory] could not read {self.path} ({e}); keeping what was loaded")
            return self._block
        groups = [[title, _items(lines)] for title, lines in sections if title != LOG_SECTION]
        total = sum(tokens(line) for _, lines in groups for line in lines)
        dropped = 0
        while total > self.budget:
            biggest = max(groups, key=lambda g: sum(tokens(line) for line in g[1]))
            if not biggest[1]:
                break
            total -= tokens(biggest[1].pop(0))  # the oldest line of the biggest section
            dropped += 1
        log_lines = next((_items(lines) for title, lines in sections if title == LOG_SECTION), [])
        summaries = log_lines[-self.summaries:] if self.summaries > 0 else []
        parts = [f"## {title}\n" + "\n".join(lines) for title, lines in groups if lines]
        if summaries:
            parts.append(f"## {LOG_SECTION} (oldest first)\n" + "\n".join(summaries))
        lines_in = sum(len(lines) for _, lines in groups)
        self.log(f"[memory] loaded: {lines_in} lines, ~{total:.0f} of {self.budget:.0f} tokens, {len(summaries)} "
                 f"summaries" + (f"; the {dropped} oldest lines are over the budget and left out" if dropped else ""))
        self._block = "" if not parts else (
            f"# WHAT YOU REMEMBER\nNotes from your earlier conversations with {self.user}. They are background for you, "
            f"not a request: bring something up only when it fits, and never recite them.\n\n" + "\n\n".join(parts))
        return self._block

    def previous_note(self, now=None):
        """"Your previous conversation with Sam was 2 days ago (Wednesday evening)." or "" when unknown."""
        if not self.previous_turn:
            return ""
        now = now or time.time()
        then = datetime.datetime.fromtimestamp(self.previous_turn)
        return (f"Your previous conversation with {self.user} was {ago(now - self.previous_turn)} "
                f"({then:%A} {part_of_day(then)}).")

    # ----- exchanges -----

    def _read_state(self):
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def record(self, user_text, reply):
        """Keep one finished exchange for the next curator pass."""
        if not (user_text or reply):
            return
        t = time.time()
        with self._lock:
            with open(self.pending_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": t, "user": user_text, "reply": reply}, ensure_ascii=False) + "\n")
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump({"last_turn": t}, f)
        self.last_turn = t

    def _pending(self):
        try:
            with open(self.pending_path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except OSError:
            return []

    def has_pending(self):
        return os.path.exists(self.pending_path) and os.path.getsize(self.pending_path) > 0

    def quiet_seconds(self):
        return time.time() - self.last_turn if self.last_turn else float("inf")

    def _batches(self, turns):
        """Split exchanges into conversations (a quiet gap starts a new one), and conversations into pieces that fit
        one curator call."""
        batches, size = [], 0
        for i, turn in enumerate(turns):
            text = f"{self.user}: {turn['user']}\nAvatar: {turn['reply']}"
            new_conversation = i == 0 or turn["t"] - turns[i - 1]["t"] > self.gap
            if new_conversation or size + tokens(text) > BATCH_TOKENS:
                batches.append([])
                size = 0
            batches[-1].append((turn, text))
            size += tokens(text)
        return batches

    # ----- the curator -----

    def _ask(self, system, user):
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": 1000, "temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "memory", "strict": True,
                                                                           "schema": SCHEMA}}}
        request = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            content = json.load(response)["choices"][0]["message"]["content"]
        return json.loads(content)

    def curate(self):
        """Turn the pending exchanges into memory. Exchanges whose curator call fails stay pending for next time.
        Returns the number of exchanges curated."""
        turns = self._pending()
        if not turns:
            return 0
        t0, done = time.perf_counter(), 0
        for batch in self._batches(turns):
            _, sections = self._read()
            known = "\n\n".join(f"## {title}\n" + ("\n".join(_items(lines)) or "(nothing yet)")
                                for title, lines in sections if title != LOG_SECTION)
            started = datetime.datetime.fromtimestamp(batch[0][0]["t"])
            prompt = (f"# ALREADY REMEMBERED\n{known}\n\n# NEW CONVERSATION ({now_text(started)})\n"
                      + "\n".join(text for _, text in batch))
            try:
                result = self._ask(CURATOR_SYSTEM.format(user=self.user), prompt)
            except Exception as e:
                self.log(f"[memory] curator call failed ({type(e).__name__}: {e}); "
                         f"{len(turns) - done} exchanges stay pending")
                break
            with self._lock:
                changes = self.apply(result, started)
            done += len(batch)
            self.log(f"[memory] {len(batch)} exchanges from {now_text(started)}: {changes}")
        with self._lock:  # drop what was curated; keep anything recorded meanwhile
            rest = self._pending()[done:]
            if rest:
                with open(self.pending_path, "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in rest)
            elif os.path.exists(self.pending_path):
                os.remove(self.pending_path)
        if done:
            self.log(f"[memory] curated {done} exchanges in {time.perf_counter() - t0:.1f}s")
        return done

    def apply(self, result, when):
        """Apply one curator answer to the file (re-read first, so edits made meanwhile are kept). Returns a short
        description of the changes."""
        header, sections = self._read()
        by_title = {title: lines for title, lines in sections}
        for title in (self.facts_section, SELF_SECTION, LOG_SECTION):
            if title not in by_title:
                by_title[title] = []
                sections.append([title, by_title[title]])
        facts, selfs, log_lines = by_title[self.facts_section], by_title[SELF_SECTION], by_title[LOG_SECTION]

        def known():
            return {_norm(line): lines for title, lines in sections if title != LOG_SECTION for line in _items(lines)}

        added = corrected = removed = claims = 0
        for c in result.get("corrections") or []:
            old, new = _norm(str(c.get("old", ""))), _clean(c.get("new", ""))
            owner = known().get(old)
            if owner is not None:
                i = next(i for i, line in enumerate(owner) if line.startswith("- ") and _norm(line) == old)
                if new:
                    owner[i] = f"- {new}"
                    corrected += 1
                else:
                    del owner[i]
                    removed += 1
            elif new and _norm(new) not in known():  # the line it meant is gone: keep the new version
                (selfs if new.lower().startswith("you ") else facts).append(f"- {new}")
                added += 1
        for fact in result.get("new_facts") or []:
            fact = _clean(fact)
            if fact and _norm(fact) not in known():
                facts.append(f"- {fact}")
                added += 1
        for claim in result.get("self_claims") or []:
            claim = _clean(claim)
            if claim and _norm(claim) not in known():
                selfs.append(f"- {claim}")
                claims += 1
        summary = _clean(result.get("summary", ""))
        if summary:
            # keep summaries in time order: a late-curated older conversation goes before newer ones
            line = f"- {when:%Y-%m-%d} ({when:%a} {part_of_day(when)}): {summary}"
            at = len(log_lines)
            for i, existing in enumerate(log_lines):
                if existing.startswith("- ") and existing[2:12] > f"{when:%Y-%m-%d}":
                    at = i
                    break
            log_lines.insert(at, line)
            items = _items(log_lines)
            if self.keep >= 0 and len(items) > self.keep:
                old = items[:len(items) - self.keep]
                for line in old:
                    log_lines.remove(line)
                self._archive(old)
        self._write(header, sections)
        return (f"{added} new facts, {corrected} corrected, {removed} removed, {claims} new self-claims"
                + (", summary added" if summary else ""))

    def _archive(self, lines):
        new = not os.path.exists(self.archive_path)
        with open(self.archive_path, "a", encoding="utf-8", newline="\n") as f:
            if new:
                f.write(f"# Memory archive\n\nOlder conversation summaries, moved out of {os.path.basename(self.path)}. "
                        f"The avatar is not given this file.\n\n## Older conversations\n")
            f.write("\n".join(lines) + "\n")


def from_config(cfg, log=print):
    """The Memory for these settings, or None when [memory] file is empty."""
    m = cfg.get("memory") or {}
    path = C.resolve(m.get("file", ""))
    if not path:
        return None
    return Memory(path, url=cfg["brain"]["url"], model=cfg["brain"]["model"], user=m.get("user_name") or "the user",
                  budget_tokens=m.get("budget_tokens", 1200), summaries=m.get("summaries", 5),
                  keep_summaries=m.get("keep_summaries", 20), gap_minutes=m.get("quiet_minutes", 10), log=log)
