"""Memory file handling, without an LLM: the curator's answer is canned.

    python tests\\test_memory.py
"""
import datetime
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from portravix import memory as M  # noqa: E402


def make(tmp, **kw):
    logs = []
    mem = M.Memory(os.path.join(tmp, "memory.md"), user="Sam", state_dir=tmp, log=logs.append, **kw)
    mem.logs = logs
    return mem


def text(mem):
    with open(mem.path, encoding="utf-8") as f:
        return f.read()


def canned(mem, *answers):
    """Make the curator return these answers in turn."""
    queue = list(answers)
    mem._ask = lambda system, user: queue.pop(0)


def answer(summary="", facts=(), corrections=(), claims=()):
    return {"summary": summary, "new_facts": list(facts), "corrections": list(corrections), "self_claims": list(claims)}


def test_new_file_has_the_sections():
    with tempfile.TemporaryDirectory() as tmp:
        mem = make(tmp)
        t = text(mem)
        for heading in ("## About Sam", "## Things you've said about yourself", "## Recent conversations"):
            assert heading in t, heading
        assert mem.block() == ""  # nothing to give the LLM yet


def test_curate_adds_and_clears_pending():
    with tempfile.TemporaryDirectory() as tmp:
        mem = make(tmp)
        mem.record("My dog is called Biscuit.", "Biscuit is a great name!")
        assert mem.has_pending()
        canned(mem, answer("Talked about Sam's dog.", ["Sam's dog is called Biscuit."], [], ["You love dogs."]))
        assert mem.curate() == 1
        assert not mem.has_pending()
        t = text(mem)
        assert "- Sam's dog is called Biscuit." in t and "- You love dogs." in t
        assert "Talked about Sam's dog." in t
        block = mem.block()
        assert block.startswith("# WHAT YOU REMEMBER") and "Biscuit" in block and "You love dogs" in block


def test_duplicates_and_corrections():
    with tempfile.TemporaryDirectory() as tmp:
        mem = make(tmp)
        when = datetime.datetime.now()
        mem.apply(answer("", ["Sam lives in Lisbon.", "Sam plays the cello."], [], ["You prefer tea."]), when)
        mem.apply(answer("", ["SAM lives in Lisbon"]), when)  # same fact, loosely
        assert text(mem).count("lives in Lisbon") == 1
        mem.apply(answer("", [], [{"old": "Sam plays the cello.", "new": "Sam plays the viola."},
                                  {"old": "You prefer tea.", "new": ""},
                                  {"old": "a line that is not there", "new": "Sam works nights."}]), when)
        t = text(mem)
        assert "viola" in t and "cello" not in t, t
        assert "You prefer tea" not in t
        assert "- Sam works nights." in t  # its old line was gone: the new version is kept


def test_edits_by_others_survive():
    with tempfile.TemporaryDirectory() as tmp:
        mem = make(tmp)
        with open(mem.path, "a", encoding="utf-8") as f:
            f.write("## Inside jokes\nKeeper's note: keep these short.\n- The cursed toaster.\n")
        mem.apply(answer("A chat.", ["Sam likes rain."]), datetime.datetime.now())
        t = text(mem)
        assert "## Inside jokes" in t and "Keeper's note: keep these short." in t and "- The cursed toaster." in t
        assert t.count(M.HEADER_MARK) == 1  # the size note is replaced, not repeated
        assert "The cursed toaster" in mem.block()  # extra sections reach the LLM too


def test_budget_keeps_the_newest_lines():
    with tempfile.TemporaryDirectory() as tmp:
        mem = make(tmp, budget_tokens=100)
        facts = [f"Sam fact number {i:02d} is a fairly ordinary sentence." for i in range(30)]
        mem.apply(answer("", facts), datetime.datetime.now())
        block = mem.block()
        assert "number 29" in block and "number 00" not in block
        assert any("over the budget" in line for line in mem.logs)
        assert "number 00" in text(mem)  # nothing is removed from the file


def test_summaries_in_prompt_and_archive():
    with tempfile.TemporaryDirectory() as tmp:
        mem = make(tmp, summaries=2, keep_summaries=3)
        start = datetime.datetime(2026, 10, 1, 19, 0)
        for day in range(5):
            mem.apply(answer(f"Conversation {day}."), start + datetime.timedelta(days=day))
        t = text(mem)
        assert "Conversation 4." in t and "Conversation 2." in t and "Conversation 1." not in t
        with open(mem.archive_path, encoding="utf-8") as f:
            archived = f.read()
        assert "Conversation 0." in archived and "Conversation 1." in archived
        block = mem.block()
        assert "Conversation 4." in block and "Conversation 3." in block and "Conversation 2." not in block
        assert "2026-10-05 (Mon evening)" in t


def test_late_summary_goes_in_date_order():
    with tempfile.TemporaryDirectory() as tmp:
        mem = make(tmp)
        mem.apply(answer("Newer."), datetime.datetime(2026, 10, 5, 9, 0))
        mem.apply(answer("Older, curated late."), datetime.datetime(2026, 10, 3, 9, 0))
        t = text(mem)
        assert t.index("Older, curated late.") < t.index("Newer.")


def test_quiet_gap_splits_conversations():
    with tempfile.TemporaryDirectory() as tmp:
        mem = make(tmp, gap_minutes=10)
        now = time.time()
        turns = [{"t": now - 7200, "user": "a", "reply": "b"}, {"t": now - 7100, "user": "c", "reply": "d"},
                 {"t": now - 60, "user": "e", "reply": "f"}]
        assert [len(b) for b in mem._batches(turns)] == [2, 1]


def test_failed_call_keeps_pending():
    with tempfile.TemporaryDirectory() as tmp:
        mem = make(tmp)
        mem.record("hello", "hi")

        def fail(system, user):
            raise OSError("server down")
        mem._ask = fail
        assert mem.curate() == 0
        assert mem.has_pending()


def test_previous_conversation_note():
    with tempfile.TemporaryDirectory() as tmp:
        mem = make(tmp)
        assert mem.previous_note() == ""
        mem.record("hi", "hello")
        later = make(tmp)  # the next session
        note = later.previous_note(now=time.time() + 2 * 86400)
        assert note.startswith("Your previous conversation with Sam was 2 days ago ("), note
    assert M.ago(30) == "1 minute ago" and M.ago(3 * 3600) == "3 hours ago" and M.ago(21 * 86400) == "3 weeks ago"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
