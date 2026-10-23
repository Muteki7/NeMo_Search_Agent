"""Build one self-contained HTML report from eval dumps: where does each model lose its points, and why?

Run it with the SERVER venv (it needs chromadb and the helpers in ../app.py), from the repository root:

    NeMo_Gym/resources_servers/mission_search/.venv/bin/python NeMo_Gym/resources_servers/mission_search/scripts/report.py \\
        --run-dir trainer/runs/20261021-141800_run2_retrieval \\
        --model "base=trainer/runs/20261021-141800_run2_retrieval/eval_base.jsonl" \\
        --model "run 1 (answer only)=trainer/runs/20261015-141100_first200/eval_checkpoint-200.jsonl" \\
        --model "run 2 (+ retrieval)=trainer/runs/20261021-141800_run2_retrieval/eval_checkpoint-200.jsonl"

Without --model, every eval_*.jsonl in --run-dir is used. The report is written to <run-dir>/report.html and opens in any browser
(no server, no internet). Up to four models: a model labelled "base..." is drawn in grey (context), the others in blue, orange, aqua.

How a trajectory gets its category (rules, first match wins; the page repeats this with the definitions):
  signals   lexical hit    : one search result holds >= 60% of the reference's content words (word overlap)
            provenance hit : a retrieved chunk overlaps the 3-chunk window the question was written from
            source support : share of the reference's content words found in that source window
  This is a set of rules, not ground truth. The category names say "suspect" where the rule can be wrong.
Needs for the full set of signals: the Chroma index (provenance), Step2_Files/Saved_data/chunks.pkl (source support, git-ignored)
and data/<split>.jsonl (the chunk_id of each question). Without chunks.pkl, category 4 is switched off.
"""
import argparse
import html
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER_DIR = HERE.parent
REPO_ROOT = SERVER_DIR.parents[2]
sys.path.insert(0, str(SERVER_DIR))
from app import STOP_WORDS, contains_run, coverage, normalize, remove_reasoning  # noqa: E402

HIT_COVERAGE = 0.6      # the same threshold as the `retrieved` metric of eval.py
SUPPORT_MIN = 0.6       # below this, the reference is poorly supported by its own source window
PARTIAL_MIN = 0.2       # right evidence and an answer reward of at least this: "partial credit"
NO_ANSWER_ENDS = ("turn_cap", "budget", "bad_call", "truncated")

# id, short name (chart), title, one-line rule, longer explanation
CATEGORIES = [
    (1, "Success", "Success",
     "Answer reward is 1.0.",
     "After the reasoning is removed, the final answer equals the reference or contains it as a run of whole words (and is short enough)."),
    (2, "No usable answer", "No usable final answer",
     "The trajectory ended on a turn cap, a budget stop, a bad tool call or a cut-off turn (or has no final message).",
     "The model never gave a normal final answer. A search on the last turn is not run, so nothing can follow it. A bad call or a cut-off turn leaves "
     "only broken text, which is scored as it is (usually near 0)."),
    (3, "Answer only in reasoning", "Answer only inside <reasoning>",
     "The reference words are in the reply, but only inside the reasoning tags.",
     "The verifier removes the reasoning before it scores, so the answer is invisible to it. This was the base model's biggest loss: a format failure, "
     "not a knowledge failure."),
    (4, "Suspect bad reference", "Suspect bad reference",
     "Fewer than 60% of the reference's content words appear in the source window it was written from.",
     "The question may be unanswerable or the reference unreliable (for example \"Not specified\"). The reference was written by an 8B model and "
     "never checked. The check is lexical, so a reference that paraphrases its source also lands here."),
    (5, "Retrieval miss", "Retrieval miss",
     "No retrieved chunk overlaps the source window, and no search result holds 60% of the reference's words.",
     "The evidence never reached the model. Nothing the model writes afterwards can fix it, and in these cases the model almost never answers right."),
    (6, "False-hit suspect", "False-hit suspect (words match, wrong place)",
     "A search result holds 60% or more of the reference's words, but no retrieved chunk overlaps the source window.",
     "The word-overlap signal says \"found\", the provenance signal says \"not from the right place\". Either the words match by chance in another passage, "
     "or the same fact is repeated elsewhere in the report."),
    (7, "Right evidence, partial credit", "Right evidence, partial credit",
     "A retrieved chunk overlaps the source window, and the answer reward is between 0.2 and 1.",
     "The model had the right text. Typical causes: a correct answer worded differently from the reference (the verifier is lexical), "
     "or an answer that is half right."),
    (8, "Right evidence, wrong answer", "Right evidence, wrong answer",
     "A retrieved chunk overlaps the source window, and the answer reward is below 0.2.",
     "The model misread the text or answered a different question. The window is three chunks wide, so the exact fact may sit in a neighbouring chunk "
     "that was not retrieved: some of these are still retrieval problems."),
]
CAT = {c[0]: c for c in CATEGORIES}


# ---------------------------------------------------------------------------------------------------------------
# Context: chunk table, source windows, chunk ids
# ---------------------------------------------------------------------------------------------------------------
class Context:
    """`data_rows` are the rows of data/<split>.jsonl as (question, chunk_id), in file order. Eval dumps carry `index`, the position
    in that file, so rows are joined by index. The question TEXT is not a key: 12% of the rows repeat a question (overlapping windows)."""

    def __init__(self, table: dict, data_rows: list, old_chunks=None):
        self.table = table              # chunk text -> (chunk_start, chunk_end), from the index
        self.data_rows = data_rows      # [(question, chunk_id)]
        self.old_chunks = old_chunks    # list of chunk texts before the merge, or None
        texts = {}
        for q, c in data_rows:
            texts.setdefault(q, set()).add(c)
        self._unique = {q: next(iter(c)) for q, c in texts.items() if len(c) == 1}

    def chunk_id_of(self, row: dict):
        """The chunk id of an eval row: by its index (checked against the question text), else by a text that occurs once."""
        i = row.get("index")
        if isinstance(i, int) and 0 <= i < len(self.data_rows) and self.data_rows[i][0] == row["question"]:
            return self.data_rows[i][1]
        return self._unique.get(row["question"])

    @staticmethod
    def window(chunk_id):
        return None if chunk_id is None else (chunk_id - 1, chunk_id + 1)

    def window_text(self, chunk_id):
        w = self.window(chunk_id)
        if w is None or self.old_chunks is None:
            return None
        return "\n".join(self.old_chunks[j] for j in range(w[0], w[1] + 1) if 0 <= j < len(self.old_chunks))


def question_text(row: dict) -> str:
    """The question inside a data row (`responses_create_params.input`)."""
    user = [m["content"] for m in row["responses_create_params"]["input"] if m["role"] == "user"][-1]
    return user.rsplit("Question:", 1)[-1].strip()


def load_context(split: str, chroma_dir: Path, chunks_pkl: Path) -> Context:
    import chromadb
    got = chromadb.PersistentClient(path=str(chroma_dir)).get_collection("example_collection").get(include=["documents", "metadatas"])
    table = {d.strip(): (m["chunk_start"], m["chunk_end"]) for d, m in zip(got["documents"], got["metadatas"])}
    rows = [json.loads(line) for line in (SERVER_DIR / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    data_rows = [(question_text(r), r["chunk_id"]) for r in rows]
    old = None
    if chunks_pkl.exists():
        old = [c.page_content for c in pickle.load(open(chunks_pkl, "rb"))]
    return Context(table, data_rows, old)


# ---------------------------------------------------------------------------------------------------------------
# Analysis of one trajectory
# ---------------------------------------------------------------------------------------------------------------
def parse_results(tool_text: str) -> list:
    """Split the tool text 'Result 1:\\n<chunk>\\n------\\nResult 2:...' back into the chunk texts."""
    pieces = re.split(r"Result \d+:\n", tool_text)[1:]
    return [re.sub(r"\n?-{3,}\s*$", "", p).strip() for p in pieces]


def evidence_sentence(chunk_texts: list, reference: str) -> str:
    """The sentence (or line) of the retrieved text that holds the most content words of the reference."""
    needed = re.findall(r"[a-z0-9]+", reference.lower())
    needed = {w for w in needed if w not in STOP_WORDS}
    best, best_n = "", -1
    for text in chunk_texts:
        for s in re.split(r"(?<=[.!?])\s+|\n", text):
            n = len(needed & set(re.findall(r"[a-z0-9]+", s.lower())))
            if n > best_n and s.strip():
                best, best_n = s.strip(), n
    return best[:420]


def categorize(reward: float, ended: str, final_text, reasoning_only: bool, support, prov: bool, lex: bool) -> int:
    if reward >= 0.999:
        return 1
    if final_text is None or ended in NO_ANSWER_ENDS:
        return 2
    if reasoning_only:
        return 3
    if support is not None and support < SUPPORT_MIN:
        return 4
    if not prov and not lex:
        return 5
    if not prov and lex:
        return 6
    return 7 if reward >= PARTIAL_MIN else 8


def analyse(row: dict, ctx: Context) -> dict:
    question, reference = row["question"], row["answer"]
    results = [o["output"] for o in row["output"] if o["type"] == "function_call_output" and isinstance(o.get("output"), str)]
    queries = []
    for c in row["output"]:
        if c["type"] == "function_call":
            try:
                queries.append(json.loads(c["arguments"]).get("query", ""))
            except (ValueError, AttributeError):
                queries.append("")
    chunk_id = ctx.chunk_id_of(row)
    window = ctx.window(chunk_id)
    chunks = []
    for s, res in enumerate(results, start=1):
        for text in parse_results(res):
            span = ctx.table.get(text.strip())
            in_window = bool(span and window and span[0] <= window[1] and span[1] >= window[0])
            chunks.append({"search": s, "text": text, "span": span, "in_window": in_window})
    lex = any(coverage(res, reference) >= HIT_COVERAGE for res in results)
    prov = any(c["in_window"] for c in chunks) if window else lex     # without a chunk id, fall back to the word-overlap signal
    window_text = ctx.window_text(chunk_id)
    support = coverage(window_text, reference) if window_text is not None else None
    final = row.get("final_text")
    ref_tokens = normalize(reference).split()
    # the tags become spaces: normalize() deletes punctuation without a space, so "<reasoning>To" would fuse into "reasoningto"
    raw_has = bool(final) and contains_run(normalize(re.sub(r"</?reasoning>", " ", final)).split(), ref_tokens)
    strip_has = bool(final) and contains_run(normalize(remove_reasoning(final)).split(), ref_tokens)
    reward = float(row["reward"])
    return {
        "chunk_id": chunk_id, "reference": reference, "reward": reward, "reward_total": row.get("reward_total"), "ended": row["ended"], "final_text": final,
        "queries": queries, "chunks": chunks, "lex": lex, "prov": prov, "support": support, "contains": strip_has,
        "reasoning_only": raw_has and not strip_has, "n_searches": len(results),
        "evidence": evidence_sentence([c["text"] for c in chunks], reference) if chunks else "",
        "category": categorize(reward, row["ended"], final, raw_has and not strip_has, support, prov, lex),
    }


def build(models: list, ctx: Context) -> tuple:
    """models: [(label, rows)]. Returns (questions, counts). Rows are aligned by `index` (their position in the data file), so a
    repeated question text is two separate questions. Only indexes present in every model's dump are kept."""
    def keyed(rows):
        return {r.get("index", pos): r for pos, r in enumerate(rows)}
    by_model = [keyed(rows) for _, rows in models]
    order = [k for k in by_model[0] if all(k in d for d in by_model)]
    questions = []
    for k in order:
        first = by_model[0][k]
        if any(d[k]["question"] != first["question"] for d in by_model):
            raise SystemExit(f"index {k} is a different question in different dumps: they are not from the same split and order")
        per_model = [analyse(d[k], ctx) for d in by_model]
        cid = per_model[0]["chunk_id"]
        questions.append({"index": k, "question": first["question"], "reference": first["answer"], "chunk_id": cid,
                          "window": ctx.window(cid), "per_model": per_model})
    counts = [{c[0]: 0 for c in CATEGORIES} for _ in models]
    for qq in questions:
        for m, a in enumerate(qq["per_model"]):
            counts[m][a["category"]] += 1
    return questions, counts


# ---------------------------------------------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------------------------------------------
esc = html.escape


def highlight(text: str, reference: str) -> str:
    """HTML-escaped `text` with the reference's content words wrapped in <mark>."""
    needed = {w for w in re.findall(r"[a-z0-9]+", reference.lower()) if w not in STOP_WORDS}
    out, last = [], 0
    for m in re.finditer(r"[A-Za-z0-9]+", text):
        out.append(esc(text[last:m.start()]))
        out.append(f"<mark>{esc(m.group())}</mark>" if m.group().lower() in needed else esc(m.group()))
        last = m.end()
    out.append(esc(text[last:]))
    return "".join(out)


def pct(n, total):
    return f"{100 * n / total:.0f}%" if total else "-"


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
--m0:#898781;--m1:#2a78d6;--m2:#eb6834;--m3:#1baf7a;--hover:rgba(11,11,11,.05);
--h0:#f0efec;--h1:#cde2fb;--h2:#b7d3f6;--h3:#9ec5f4;--h4:#86b6ef;--h5:#6da7ec;--h6:#3987e5;--h7:#2a78d6;--h8:#256abf;--h9:#1c5cab;--hink:#0b0b0b;--hwhite:#fff;}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--m1:#3987e5;--m2:#d95926;--m3:#199e70;--hover:rgba(255,255,255,.07);
--h0:#252523;--h1:#0d366b;--h2:#104281;--h3:#184f95;--h4:#1c5cab;--h5:#256abf;--h6:#3987e5;--h7:#5598e7;--h8:#6da7ec;--h9:#86b6ef;--hink:#f0efec;--hwhite:#0b0b0b;}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--m1:#3987e5;--m2:#d95926;--m3:#199e70;--hover:rgba(255,255,255,.07);
--h0:#252523;--h1:#0d366b;--h2:#104281;--h3:#184f95;--h4:#1c5cab;--h5:#256abf;--h6:#3987e5;--h7:#5598e7;--h8:#6da7ec;--h9:#86b6ef;--hink:#f0efec;--hwhite:#0b0b0b;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px}h3{font-size:15px;margin:0 0 8px}h4{font-size:14px;margin:16px 0 6px}
.sub,.note{color:var(--ink2)}.note{font-size:13px}.muted{color:var(--muted)}
.card-box{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:0 0 12px;font-size:13px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:6px}.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.m0{background:var(--m0)}.m1{background:var(--m1)}.m2{background:var(--m2)}.m3{background:var(--m3)}
.chart{position:relative}.crow{display:grid;grid-template-columns:minmax(120px,210px) 1fr;gap:12px;align-items:center;padding:7px 0;border-top:1px solid var(--grid)}
.crow:first-child{border-top:0}.clabel{font-size:13px;color:var(--ink);line-height:1.25}
.bars{position:relative;min-height:24px;display:flex;flex-direction:column;justify-content:center;gap:2px}
.brow{display:flex;align-items:center;gap:6px;height:14px}
.bar{height:14px;border-radius:0 4px 4px 0;min-width:0;flex:none}.bar:focus-visible,.bar:hover{outline:2px solid var(--ink);outline-offset:1px}
.val{font-size:12px;color:var(--ink2)}
.axis{display:grid;grid-template-columns:minmax(120px,210px) 1fr;gap:12px;border-top:1px solid var(--axis);padding-top:4px}
.ticks{position:relative;height:18px;font-size:11px;color:var(--muted)}.ticks span{position:absolute;transform:translateX(-50%)}
button,select,input[type=search]{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--axis);border-radius:8px;padding:6px 10px}
button{cursor:pointer}button:hover{background:var(--hover)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 10px;border-top:1px solid var(--grid)}th{color:var(--ink2);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.defs dt{font-weight:600;margin-top:12px}.defs dd{margin:2px 0 0;color:var(--ink2)}.defs .rule{color:var(--ink)}
.filters{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center;margin:0 0 16px}.filters label{display:inline-flex;gap:6px;align-items:center;font-size:13px;color:var(--ink2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:12px 14px}.tile .l{font-size:13px;color:var(--ink2)}.tile .v{font-size:28px;font-weight:600;line-height:1.2}.tile .s{font-size:12px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:auto 1fr 1fr;gap:2px;max-width:560px;font-size:13px}.grid2 .hd{color:var(--ink2);padding:6px 8px;align-self:center}
.cell{padding:12px 10px;border-radius:6px;text-align:center}.cell b{display:block;font-size:22px;font-weight:600}.cell small{font-size:12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;margin:0 0 8px}.card[hidden]{display:none}
.card>details>summary{cursor:pointer;padding:10px 14px;list-style:none}.card>details>summary::-webkit-details-marker{display:none}
.card>details>summary:hover{background:var(--hover);border-radius:10px}.qline{font-size:14px}.qid{color:var(--muted);margin-right:8px;font-variant-numeric:tabular-nums}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}.chip{display:inline-flex;align-items:center;gap:6px;font-size:12px;border:1px solid var(--ring);border-radius:999px;padding:1px 9px;color:var(--ink2)}
.chip i{width:8px;height:8px;border-radius:50%;display:inline-block}
.body{padding:2px 14px 14px;border-top:1px solid var(--grid)}.body p{margin:6px 0}.kv{display:grid;grid-template-columns:150px 1fr;gap:3px 12px;font-size:13px}.kv dt{color:var(--ink2)}.kv dd{margin:0}
.mblock{border-left:4px solid var(--grid);padding:2px 0 2px 12px;margin-top:14px}.mblock.m0{border-left-color:var(--m0);background:none}.mblock.m1{border-left-color:var(--m1);background:none}.mblock.m2{border-left-color:var(--m2);background:none}.mblock.m3{border-left-color:var(--m3);background:none}
pre{white-space:pre-wrap;word-break:break-word;background:var(--page);border:1px solid var(--grid);border-radius:8px;padding:8px 10px;font:12.5px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;margin:6px 0}
mark{background:rgba(237,161,0,.35);color:inherit;border-radius:2px;padding:0 1px}
details.sub details{margin:4px 0}summary{cursor:pointer}
.tip{position:fixed;z-index:10;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--axis);border-radius:8px;padding:6px 10px;font-size:12.5px;box-shadow:0 2px 8px rgba(0,0,0,.18);max-width:300px}
.tip b{font-size:15px}
.cell{color:var(--hink)}.cell.w{color:var(--hwhite)}
.h0{background:var(--h0)}.h1{background:var(--h1)}.h2{background:var(--h2)}.h3{background:var(--h3)}.h4{background:var(--h4)}.h5{background:var(--h5)}.h6{background:var(--h6)}.h7{background:var(--h7)}.h8{background:var(--h8)}.h9{background:var(--h9)}
@media (max-width:640px){.crow,.axis{grid-template-columns:1fr}.kv{grid-template-columns:1fr}}
"""

JS = """
(function(){
  var $=function(s,r){return (r||document).querySelector(s)}, $$=function(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s))};
  var tip=document.createElement('div'); tip.className='tip'; tip.hidden=true; document.body.appendChild(tip);
  function show(el,x,y){ // tooltip text comes from data attributes; build it with textContent only
    tip.textContent=''; var v=document.createElement('b'); v.textContent=el.getAttribute('data-v'); tip.appendChild(v);
    var l=document.createElement('div'); l.textContent=el.getAttribute('data-l'); tip.appendChild(l);
    tip.hidden=false; tip.style.left=Math.min(x+14,window.innerWidth-320)+'px'; tip.style.top=(y+14)+'px';}
  $$('.bar').forEach(function(b){
    b.addEventListener('pointermove',function(e){show(b,e.clientX,e.clientY)});
    b.addEventListener('pointerleave',function(){tip.hidden=true});
    b.addEventListener('focus',function(){var r=b.getBoundingClientRect();show(b,r.right,r.top)});
    b.addEventListener('blur',function(){tip.hidden=true});});
  var tv=$('#toggle-table'); if(tv){tv.addEventListener('click',function(){var t=$('#chart-table'),c=$('#chart-bars'),on=t.hidden;
    t.hidden=!on; c.hidden=on; tv.textContent=on?'Show chart':'Show as table'; tv.setAttribute('aria-pressed',on);});}
  var model=$('#f-model'), cat=$('#f-cat'), q=$('#f-search'), diff=$('#f-diff'), count=$('#f-count');
  function apply(){
    var m=parseInt(model.value,10), c=cat.value, s=q.value.trim().toLowerCase(), d=diff.checked, n=0, cards=$$('.card');
    $$('.panel').forEach(function(p){p.hidden=(p.getAttribute('data-model')!==String(m))});
    cards.forEach(function(card){
      var cats=card.getAttribute('data-cats').split(','), ok=true;
      if(c!=='all' && cats[m]!==c) ok=false;
      if(s && card.getAttribute('data-search').indexOf(s)<0) ok=false;
      if(d && cats.every(function(x){return x===cats[0]})) ok=false;
      card.hidden=!ok; if(ok) n++;});
    count.textContent=n+' of '+cards.length+' questions';}
  [model,cat,diff].forEach(function(el){el.addEventListener('change',apply)}); q.addEventListener('input',apply); apply();
})();
"""


def model_class(label: str, index: int, slot_counter: list) -> str:
    if label.lower().startswith("base"):
        return "m0"
    slot_counter[0] += 1
    return f"m{slot_counter[0]}"


def nice_max(n: int) -> int:
    step = 10 if n > 20 else 5
    return max(step, ((n + step - 1) // step) * step)


def chart_html(labels, classes, counts, total) -> str:
    axis_max = nice_max(max(max(c.values()) for c in counts))
    steps = axis_max // (10 if axis_max > 20 else 5)
    ticks = list(range(0, axis_max + 1, axis_max // steps))
    legend = "".join(f'<span><i class="sw {cl}"></i>{esc(lb)}</span>' for lb, cl in zip(labels, classes))
    rows = []
    for cid, short, title, rule, _ in CATEGORIES:
        bars = []
        for m, (lb, cl) in enumerate(zip(labels, classes)):
            n = counts[m][cid]
            tip_l = f"{lb}: {title}"
            label_val = f'<span class="val">{n}</span>' if m == len(labels) - 1 else ""
            bar = (f'<div class="bar {cl}" style="width:{100 * n / axis_max:.2f}%" tabindex="0" data-v="{n} of {total} ({pct(n, total)})" '
                   f'data-l="{esc(tip_l)}" aria-label="{esc(tip_l)}: {n} of {total}"></div>') if n else '<div class="bar" style="width:0"></div>'
            bars.append(f'<div class="brow">{bar}{label_val}</div>')
        rows.append(f'<div class="crow"><div class="clabel" title="{esc(rule)}">{cid}. {esc(short)}</div><div class="bars" style="background-image:linear-gradient(to right,var(--grid) 1px,transparent 1px);background-size:{100 * (axis_max // steps) / axis_max:.4f}% 100%">{"".join(bars)}</div></div>')
    tick_html = "".join(f'<span style="left:{100 * t / axis_max}%">{t}</span>' for t in ticks)
    head = "".join(f"<th class='n'>{esc(lb)}</th>" for lb in labels)
    trs = "".join(f"<tr><td>{cid}. {esc(title)}</td>" + "".join(f"<td class='n'>{counts[m][cid]} ({pct(counts[m][cid], total)})</td>" for m in range(len(labels))) + "</tr>"
                  for cid, _, title, _, _ in CATEGORIES)
    return (f'<div class="legend">{legend}</div>'
            f'<div id="chart-bars" class="chart">'
            f'{"".join(rows)}<div class="axis"><div></div><div class="ticks">{tick_html}</div></div></div>'
            f'<div id="chart-table" hidden><table><thead><tr><th>Category</th>{head}</tr></thead><tbody>{trs}</tbody></table></div>')


def tiles_and_grid(m: int, labels, questions, counts, total) -> str:
    A = [q["per_model"][m] for q in questions]
    exact = counts[m][1]
    tiles = [
        ("Exact answers", exact, "reward 1.0"),
        ("Right chunk retrieved", sum(a["prov"] for a in A), "provenance hit"),
        ("Words found in results", sum(a["lex"] for a in A), "lexical hit"),
        ("Answer in the final text", sum(a["contains"] for a in A), "whatever its length"),
    ]
    tile_html = "".join(f'<div class="tile"><div class="l">{esc(l)}</div><div class="v">{v} <span class="muted" style="font-size:15px;font-weight:400">{pct(v, total)}</span></div><div class="s">of {total} questions · {esc(s)}</div></div>'
                        for l, v, s in tiles)
    cells = {}
    for a in A:
        cells.setdefault((a["lex"], a["prov"]), []).append(a)
    peak = max([len(v) for v in cells.values()] + [1])

    def cell(lex, prov):
        items = cells.get((lex, prov), [])
        n = len(items)
        level = 0 if n == 0 else 1 + min(8, int(8 * n / peak))
        cont = (f"{100 * sum(a['contains'] for a in items) / n:.0f}% answered" if n else "-")
        cls = f"cell h{level}" + (" w" if level >= 6 else "")
        return f'<div class="{cls}"><b>{n}</b><small>{cont}</small></div>'

    grid = ('<div class="grid2"><div></div><div class="hd">Words found (lexical hit)</div><div class="hd">Words not found</div>'
            f'<div class="hd">Right chunk retrieved (provenance hit)</div>{cell(True, True)}{cell(False, True)}'
            f'<div class="hd">Right chunk not retrieved</div>{cell(True, False)}{cell(False, False)}</div>'
            '<p class="note">Counts of questions; the second line is how often the final answer contained the reference. Darker means more questions.</p>')
    return f'<div class="panel" data-model="{m}"><div class="tiles">{tile_html}</div><h3>Do the two retrieval signals agree?</h3>{grid}</div>'


def card_html(i: int, q: dict, labels, classes) -> str:
    cats = ",".join(str(a["category"]) for a in q["per_model"])
    search = esc((q["question"] + " " + q["reference"]).lower())
    chips = "".join(f'<span class="chip"><i class="{cl}"></i>{esc(lb)}: {a["category"]}. {esc(CAT[a["category"]][1])} · {a["reward"]:.2f}</span>'
                    for lb, cl, a in zip(labels, classes, q["per_model"]))
    win = q["window"]
    head = (f'<dl class="kv"><dt>Reference answer</dt><dd><b>{esc(q["reference"])}</b></dd>'
            f'<dt>Source window</dt><dd>{"old chunks %d to %d (question written from chunk %d)" % (win[0], win[1], q["chunk_id"]) if win else "unknown (no chunk id)"}</dd>'
            f'<dt>Reference supported by its source</dt><dd>{"%.0f%% of its content words" % (100 * q["per_model"][0]["support"]) if q["per_model"][0]["support"] is not None else "not measured (chunks.pkl missing)"}</dd></dl>')
    blocks = []
    for lb, cl, a in zip(labels, classes, q["per_model"]):
        chunks = "".join(
            f'<details class="sub"><summary>Search {c["search"]}, chunk covering old positions {c["span"][0]} to {c["span"][1]}'
            f'{" (inside the source window)" if c["in_window"] else ""}</summary><pre>{highlight(c["text"], q["reference"])}</pre></details>' if c["span"] else
            f'<details class="sub"><summary>Search {c["search"]}, chunk not found in the index</summary><pre>{highlight(c["text"], q["reference"])}</pre></details>'
            for c in a["chunks"])
        queries = "; ".join(f'"{esc(x)}"' for x in a["queries"]) or "none"
        total = f' (combined training reward {a["reward_total"]:.2f})' if a["reward_total"] is not None else ""
        blocks.append(
            f'<div class="mblock {cl}"><h4>{esc(lb)}</h4><dl class="kv">'
            f'<dt>Category</dt><dd>{a["category"]}. {esc(CAT[a["category"]][2])}</dd>'
            f'<dt>Answer reward</dt><dd>{a["reward"]:.2f}{total} · ended: {esc(a["ended"])}</dd>'
            f'<dt>Searches</dt><dd>{a["n_searches"]}: {queries}</dd>'
            f'<dt>Retrieval signals</dt><dd>words found: {"yes" if a["lex"] else "no"} · right chunk: {"yes" if a["prov"] else "no"}</dd>'
            f'<dt>Best evidence line</dt><dd>{highlight(a["evidence"], q["reference"]) if a["evidence"] else "<span class=muted>nothing retrieved</span>"}</dd></dl>'
            f'{chunks}<p><b>Final answer</b></p><pre>{esc(a["final_text"]) if a["final_text"] else "(no final message)"}</pre></div>')
    return (f'<article class="card" data-cats="{cats}" data-search="{search}"><details><summary><div class="qline"><span class="qid">Q{i + 1}</span>{esc(q["question"])}</div>'
            f'<div class="chips">{chips}</div></summary><div class="body">{head}{"".join(blocks)}</div></details></article>')


def definitions_html() -> str:
    items = "".join(f'<dt>{cid}. {esc(title)}</dt><dd class="rule">{esc(rule)}</dd><dd>{esc(long)}</dd>' for cid, _, title, rule, long in CATEGORIES)
    return (
        '<p>Every question gets one category per model, by these rules in this order (the first one that matches wins). '
        'They are rules, not ground truth: where a rule can be wrong the name says "suspect". Open a question below to check it by eye.</p>'
        '<dl class="defs"><dt>Lexical hit</dt><dd>One search result holds at least 60% of the reference answer\'s content words (stop words removed). '
        'A bag of words: no order, no meaning.</dd>'
        '<dt>Provenance hit</dt><dd>A retrieved chunk overlaps the three-chunk window that the question was written from. It does not depend on wording. '
        'The tool text is matched back to the index by exact text.</dd>'
        '<dt>Source support</dt><dd>The share of the reference\'s content words found in that source window.</dd>'
        f'{items}</dl>')


def page(run_name: str, labels, classes, questions, counts, files) -> str:
    total = len(questions)
    cat_opts = '<option value="all">all categories</option>' + "".join(f'<option value="{c[0]}">{c[0]}. {esc(c[1])}</option>' for c in CATEGORIES)
    model_opts = "".join(f'<option value="{i}">{esc(lb)}</option>' for i, lb in enumerate(labels))
    panels = "".join(tiles_and_grid(m, labels, questions, counts, total) for m in range(len(labels)))
    cards = "".join(card_html(i, q, labels, classes) for i, q in enumerate(questions))
    files_html = "".join(f"<li>{esc(lb)}: <code>{esc(str(p))}</code></li>" for lb, p in zip(labels, files))
    last = len(labels) - 1
    lead = (f"<b>{esc(labels[last])}</b>: {counts[last][1]} of {total} questions are exact ({pct(counts[last][1], total)}). "
            f"The biggest losses are retrieval misses ({counts[last][5]}) and right evidence with a partial or wrong answer ({counts[last][7] + counts[last][8]}).")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval report</title><style>{CSS}</style></head><body><main>
<h1>Eval report: {esc(run_name)}</h1>
<p class="sub">{total} validation questions, greedy decoding. Rewards are the <b>answer-only</b> reward, comparable across runs. Generated {datetime.now():%Y-%m-%d %H:%M}.</p>
<p>{lead}</p>
<noscript><p class="note">JavaScript is off: the filters, tooltips and the table toggle do not work, but every number is on the page.</p></noscript>

<h2>1. Where each model loses its points</h2>
<div class="card-box">
<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px">
<span class="note">Number of questions in each category (of {total}). Hover or focus a bar for the exact count.</span>
<button id="toggle-table" type="button" aria-pressed="false">Show as table</button></div>
{chart_html(labels, classes, counts, total)}
</div>

<h2>2. What the categories mean</h2>
<div class="card-box">{definitions_html()}</div>

<h2>3. One model in detail</h2>
<div class="filters" role="group" aria-label="Filters">
<label>Model <select id="f-model">{model_opts}</select></label>
<label>Category <select id="f-cat">{cat_opts}</select></label>
<label>Search <input id="f-search" type="search" placeholder="question or reference"></label>
<label><input id="f-diff" type="checkbox"> only where the models differ</label>
<span class="note" id="f-count"></span></div>
{panels}
<h2>4. The questions</h2>
<p class="note">The model, category and search filters above apply to this list. Click a question to read the evidence and the answers of every model side by side.</p>
{cards}
<h2>Sources</h2><ul class="note">{files_html}</ul>
</main><script>{JS}</script></body></html>"""


# ---------------------------------------------------------------------------------------------------------------
def load_models(args) -> tuple:
    specs = []
    if args.model:
        for s in args.model:
            label, _, path = s.partition("=")
            if not path:
                raise SystemExit(f"--model needs LABEL=PATH, got {s!r}")
            specs.append((label.strip(), Path(path)))
    else:
        def key(p):
            n = re.search(r"(\d+)", p.stem)
            return (0 if "base" in p.stem else 1, int(n.group(1)) if n else 0, p.stem)
        specs = [(p.stem.replace("eval_", ""), p) for p in sorted(args.run_dir.glob("eval_*.jsonl"), key=key)]
    if not specs:
        raise SystemExit("no models: pass --model LABEL=PATH or point --run-dir at a folder with eval_*.jsonl")
    if len(specs) > 4:
        raise SystemExit("at most 4 models (one grey base plus three colours): pick the ones to compare")
    return [(lb, [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]) for lb, p in specs], [p for _, p in specs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, help="the report goes here (report.html)")
    ap.add_argument("--model", action="append", help="LABEL=PATH to an eval dump; repeat for each model (max 4)")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--chroma-dir", type=Path, default=SERVER_DIR / "chroma_index")
    ap.add_argument("--chunks", type=Path, default=REPO_ROOT / "Step2_Files" / "Saved_data" / "chunks.pkl")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    models, files = load_models(args)
    ctx = load_context(args.split, args.chroma_dir, args.chunks)
    if ctx.old_chunks is None:
        print(f"[report] {args.chunks} not found: category 4 (suspect bad reference) is switched off")
    questions, counts = build(models, ctx)
    if not questions:
        raise SystemExit("no question is present in every model's dump")
    labels = [lb for lb, _ in models]
    slots = [0]
    classes = [model_class(lb, i, slots) for i, lb in enumerate(labels)]
    out = args.out or args.run_dir / "report.html"
    out.write_text(page(args.run_dir.name, labels, classes, questions, counts, files), encoding="utf-8")
    print(f"[report] {len(questions)} questions, {len(labels)} models -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    for lb, c in zip(labels, counts):
        print(f"  {lb}: " + ", ".join(f"{k}:{v}" for k, v in c.items()))


if __name__ == "__main__":
    main()
