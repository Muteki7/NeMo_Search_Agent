"""Draws the schema images for Debug_checks.md (the Step 3 index build) into this folder.

    NeMo_Gym/resources_servers/mission_search/.venv/bin/python Schema_learning/build_schema.py

The diagrams go from the whole picture (01) to the detail you fill in at the debugger (04).
Facts come from Debug_checks.md. Colors: blue = build_index.py, green = embeddings.py, gray = library code.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
BLUE, GREEN, GRAY, ORANGE = "#2B6CB0", "#2F855A", "#718096", "#DD6B20"
BLUE_L, GREEN_L, GRAY_L, ORANGE_L = "#EBF4FF", "#E6F6EF", "#EDF2F7", "#FEEBC8"
INK, MUTED = "#1A202C", "#4A5568"
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
TAG_COLOR = {"real run": GREEN, "estimate": ORANGE, "arithmetic": BLUE}


# ------------------------------------------------------------------ helpers
def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")
    ax.text(4, 85.5, title, fontsize=22, fontweight="bold", color=INK, va="center", family=SANS)
    ax.text(4, 81, subtitle, fontsize=12.5, color=MUTED, va="center", family=SANS)
    return fig, ax


def box(ax, x, y, w, h, fc="white", ec=INK, lw=1.6, r=1.1, ls="-", z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def text(ax, x, y, s, size=11, color=INK, weight="normal", ha="left", va="center", family=SANS, z=5, style="normal"):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, family=family, zorder=z, style=style)


def arrow(ax, p1, p2, color=INK, lw=2.0, rad=0.0, ls="-", z=3, ms=14):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def badge(ax, x, y, n, r=1.5, size=10.5):
    ax.add_patch(Circle((x, y), r, fc=ORANGE, ec="white", lw=1.4, zorder=6))
    text(ax, x, y - 0.05, str(n), size=size, color="white", weight="bold", ha="center", z=7)


def tag(ax, x, y, name, size=8.5, ha="right"):
    if name:
        text(ax, x, y, f"[{name}]", size=size, color=TAG_COLOR[name], ha=ha, family=MONO)


def save(fig, name):
    fig.savefig(OUT / name, dpi=120, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------ 01 big picture
def big_picture():
    fig, ax = canvas("The whole index build in one picture",
                     "Read left to right. Each numbered circle is one debugger stop (details in 04_stop_map.png).")
    W, GAP, X0, Y0, H = 27.5, 3.4, 4.0, 24.0, 42.0
    phases = [
        ("A", "PREPARE", "build_index.py : 83-93", BLUE, BLUE_L,
         ["Guard: protect a real index", "Load the 753 chunks", "Clean metadata for Chroma", "Add chunk_index to each one"],
         [(1, "Guard + globals"), (2, "Metadata cleaning"), (3, "Loaded chunks")],
         ["753 chunks,", "ready to embed"]),
        ("B", "BUILD THE EMBEDDER", "embeddings.py : 29-37", GREEN, GREEN_L,
         ["Load model + tokenizer", "Device: CPU (our choice)", "Batch size: 32 (our choice)", "Eval mode: no dropout"],
         [(4, "The embedder")],
         ["embedder object", "(model + tokenizer)"]),
        ("C", "EMBED + STORE", "embeddings.py : 66 + library", GREEN, GREEN_L,
         ["Chroma asks the embedder", "24 batches: 23 x 32 + 17", "One [CLS] vector per chunk", "Chroma writes the index"],
         [(5, "Chroma calls embedder"), (6, "Batch to vectors"), (7, "Index exists")],
         ["753 vectors on disk", "folder size ~6.3 MB"]),
        ("D", "REOPEN FROM DISK", "build_index.py : 105-111", BLUE, BLUE_L,
         ["Drop the in-memory object", "Open the files again", "Row count must be 753", "Embedder: queries only"],
         [(8, "Reopen from disk")],
         ["the index works", "on its own"]),
        ("E", "TEST WITH QUESTIONS", "build_index.py : 49-120", BLUE, BLUE_L,
         ["Sample 400 of 1434 questions", "Embed each: mean pooling", "Search top 2 and top 5", "Count window hits"],
         [(9, "The sample"), (10, "One real search"), (11, "The verdict")],
         ["recall 78.5% (top 2)", "recall 88.0% (top 5)"]),
    ]
    xs = []
    for i, (letter, name, files, col, light, lines, stops, out) in enumerate(phases):
        x = X0 + i * (W + GAP)
        xs.append(x)
        box(ax, x, Y0, W, H, fc=light, ec=col, lw=2)
        box(ax, x, Y0 + H - 7.2, W, 7.2, fc=col, ec=col, lw=2)
        text(ax, x + 1.4, Y0 + H - 3.6, f"{letter}  {name}", size=11.5, color="white", weight="bold")
        text(ax, x + 1.4, Y0 + H - 10.3, files, size=8.6, color=col, family=MONO, weight="bold")
        for k, ln in enumerate(lines):
            text(ax, x + 1.4, Y0 + H - 14.6 - 3.2 * k, ln, size=10)
        ax.plot([x + 1.2, x + W - 1.2], [Y0 + 16.0, Y0 + 16.0], color=col, lw=1, alpha=0.5, zorder=3)
        text(ax, x + 1.4, Y0 + 14.0, "YOUR STOPS", size=8.3, color=MUTED, weight="bold")
        for k, (n, label) in enumerate(stops):
            yy = Y0 + 10.8 - 3.7 * k
            badge(ax, x + 3.0, yy, n)
            text(ax, x + 5.6, yy, label, size=10, weight="bold")
        # output strip
        box(ax, x, 12.5, W, 8.5, fc="white", ec=col, lw=1.6, ls=(0, (4, 2)))
        text(ax, x + W / 2, 18.4, out[0], size=10, ha="center", weight="bold")
        text(ax, x + W / 2, 15.2, out[1], size=10, ha="center")
        arrow(ax, (x + W / 2, Y0), (x + W / 2, 21.0), color=col)
        if i:
            arrow(ax, (xs[i - 1] + W, Y0 + H / 2), (x, Y0 + H / 2), color=INK, lw=2.2)

    # inputs
    for idx, (label, sub) in ((0, ("chunks.pkl", "753 Document objects")), (4, ("questions.json", "1434 questions"))):
        x = xs[idx]
        box(ax, x, 70.0, W, 6.4, fc="white", ec=INK, lw=1.5)
        text(ax, x + W / 2, 74.2, label, size=11, ha="center", weight="bold", family=MONO)
        text(ax, x + W / 2, 71.6, sub, size=9.5, ha="center", color=MUTED)
        arrow(ax, (x + W / 2, 70.0), (x + W / 2, Y0 + H), color=INK)
    # disk
    dx0, dx1 = xs[2], xs[3] + W
    box(ax, dx0, 2.0, dx1 - dx0, 6.2, fc=GRAY_L, ec=GRAY, lw=1.8)
    text(ax, (dx0 + dx1) / 2, 5.1, "chroma_index/  on disk:  chroma.sqlite3  +  one UUID folder", size=10.5, ha="center", weight="bold", family=MONO)
    arrow(ax, (xs[2] + W / 2, 12.5), (xs[2] + W / 2, 8.2), color=GRAY)
    text(ax, xs[2] + W / 2 + 1.2, 10.4, "write", size=9, color=GRAY, weight="bold")
    arrow(ax, (xs[3] + W / 2, 8.2), (xs[3] + W / 2, 12.5), color=GRAY)
    text(ax, xs[3] + W / 2 + 1.2, 10.4, "read back", size=9, color=GRAY, weight="bold")
    # legend
    lx, ly = 4.0, 6.6
    for k, (c, l, label) in enumerate(((BLUE, BLUE_L, "build_index.py (our script)"), (GREEN, GREEN_L, "embeddings.py (our embedder)"))):
        box(ax, lx, ly - 2.0 - 3.6 * k + 0.6, 2.6, 2.2, fc=l, ec=c, lw=1.6)
        text(ax, lx + 3.8, ly - 1.0 - 3.6 * k + 0.6, label, size=9.5)
    badge(ax, lx + 31.5, ly - 0.4, "n", r=1.3, size=9)
    text(ax, lx + 33.6, ly - 0.4, "= a debugger stop", size=9.5)
    box(ax, lx + 30.2, ly - 5.1, 2.6, 2.2, fc="white", ec=INK, lw=1.4, ls=(0, (4, 2)))
    text(ax, lx + 33.6, ly - 4.0, "= output of the phase", size=9.5)
    save(fig, "01_big_picture.png")


# ------------------------------------------------------------------ 02 two paths
def two_paths():
    fig, ax = canvas("Two paths through the same model",
                     "Documents are embedded when we build (Stops 5-6). Questions are embedded when we search (Stop 10).")

    def panel(x0, title, sub, color, light, steps, code, stop_marks, hl):
        W = 72.0
        box(ax, x0, 13.2, W, 64.0, fc=light, ec=color, lw=2)
        box(ax, x0, 70.2, W, 7.0, fc=color, ec=color, lw=2)
        text(ax, x0 + 2, 73.7, title, size=14, color="white", weight="bold")
        text(ax, x0 + W - 2, 73.7, sub, size=9.5, color="white", ha="right")
        tops = [66.0 - 8.2 * k for k in range(6)]
        for k, (name, shape, tg) in enumerate(steps):
            top = tops[k]
            ec = ORANGE if k == hl else color
            box(ax, x0 + 2.5, top - 6.2, W - 5, 6.2, fc="white", ec=ec, lw=3.2 if k == hl else 1.6, z=3)
            text(ax, x0 + 6.2, top - 3.1, name, size=10.8, weight="bold")
            text(ax, x0 + 37.5, top - 2.2, shape, size=9.4, family=MONO)
            tag(ax, x0 + W - 3.5, top - 4.7, tg, size=8.2)
            if k < 5:
                arrow(ax, (x0 + W / 2, top - 6.2), (x0 + W / 2, tops[k + 1]), color=color, lw=2)
        for k, n in stop_marks:
            badge(ax, x0 + 2.5, tops[k] - 0.4, n)
        text(ax, x0 + 2.5, 15.6, code, size=8.8, family=MONO, color=color, weight="bold")
        text(ax, x0 + W - 2.5, tops[hl] + 1.2, "THE ONLY DIFFERENCE", size=8.2, color=ORANGE, weight="bold", ha="right")

    panel(4, "DOCUMENT PATH", "used to build the index", GREEN, GREEN_L,
          [("753 chunk texts", "list of 753 strings", ""),
           ("Batches of 32", "24 passes: 23 x 32 + 17", "arithmetic"),
           ("Tokenizer", "input_ids (32, L)  L <= 512", "estimate"),
           ("Model", "last_hidden_state (32, L, hidden)", "estimate"),
           ("Keep the first token: [CLS]", "[:, 0, :]  ->  (32, hidden)", ""),
           ("Chroma stores the vectors", "ids chunk-0 ... chunk-752", "real run")],
          "embed_documents  embeddings.py:66-74   get_embedding(mode='sentence')  :39-64",
          [(1, 5), (2, 6), (3, 6), (4, 6), (5, 7)], 4)
    panel(84, "QUERY PATH", "used to search the index", BLUE, BLUE_L,
          [("1 question text", "400 questions in the check", "real run"),
           ("One at a time", "no batch: 1 text per call", ""),
           ("Tokenizer", "input_ids (1, L)", "estimate"),
           ("Model", "last_hidden_state (1, L, hidden)", "estimate"),
           ("Average all tokens (mask-weighted)", "mean over L  ->  (1, hidden)", ""),
           ("Chroma finds the nearest vectors", "top 2 / top 5 by distance", "real run")],
          "embed_query  embeddings.py:76-81   get_embedding(mode='query')  :39-64",
          [(0, 9), (4, 10)], 4)

    box(ax, 4, 2.0, 152, 9.2, fc=GRAY_L, ec=GRAY, lw=1.8)
    text(ax, 80, 8.6, "Same model, same tokenizer, same class. The pooling differs on purpose: this is how the original code works.",
         size=11, ha="center", weight="bold")
    text(ax, 80, 5.6, "Both vectors must come from this one class, or the distances mean nothing.  Lower distance = closer match.",
         size=10.5, ha="center")
    text(ax, 80, 3.3, "Real run (heart-rate question): chunk 364 at distance ~16.3, chunk 362 at ~22.0.   Distance setting: l2 [estimate].",
         size=10, ha="center", color=MUTED)
    save(fig, "02_two_paths.png")


# ------------------------------------------------------------------ 03 call map
def call_map():
    fig, ax = canvas("Who calls whom (this is the Call Stack panel)",
                     "Follow one path from <module> to the right. The Call Stack shows that path, newest call on top.")
    NW, NH = 17.5, 9.0
    C = [10.0, 32.8, 55.6, 78.4, 101.2, 124.0, 146.8]
    KIND = {"build": (BLUE, BLUE_L, "-"), "embed": (GREEN, GREEN_L, "-"), "lib": (GRAY, GRAY_L, "-"), "native": (GRAY, "white", (0, (4, 2)))}

    def node(col, y, name, loc, kind, badges=(), h=NH, w=NW):
        c, l, ls = KIND[kind]
        x, yb = C[col] - w / 2, y - h / 2
        box(ax, x, yb, w, h, fc=l, ec=c, lw=1.9, ls=ls)
        lines = name.split("\n")
        for k, ln in enumerate(lines):
            text(ax, C[col], y + h / 2 - 2.3 - 2.1 * k, ln, size=8.8, family=MONO, weight="bold", ha="center", color=INK)
        text(ax, C[col], y - h / 2 + 1.5, loc, size=7.8, ha="center", color=c, family=MONO)
        for j, n in enumerate(badges):
            badge(ax, x + 1.5 + 3.3 * j, y + h / 2 + 0.2, n, r=1.35, size=9)
        return (x, x + w, y)

    nd = {}
    nd["module"] = node(0, 45, "<module>", "build_index.py:124", "build")
    nd["main"] = node(1, 45, "main()", "build_index.py:74", "build", badges=(1, 7, 8, 11))
    nd["load"] = node(2, 72, "load_chunks", "build_index.py:41", "build", badges=(2, 3))
    nd["clean"] = node(3, 72, "clean_metadata", "build_index.py:33", "build", badges=(2,))
    nd["init"] = node(2, 59, "Custom...Embeddings\n.__init__", "embeddings.py:29", "embed", badges=(4,))
    nd["pre"] = node(3, 59, "from_pretrained\n(model, tokenizer)", "library + native", "native")
    nd["fd"] = node(2, 45, "Chroma\n.from_documents", "vectorstores.py:1375", "lib")
    nd["ft"] = node(3, 45, "from_texts", "vectorstores.py:1273", "lib")
    nd["at"] = node(4, 45, "add_texts", "vectorstores.py:597", "lib")
    nd["ed"] = node(5, 45, "embed_documents", "embeddings.py:66", "embed", badges=(5,))
    nd["re"] = node(2, 29, "Chroma(...)\nreopen + count", "build_index.py:107", "build")
    nd["nat"] = node(3, 29, "Chroma storage\n(native code)", "upsert + query", "native")
    nd["rc"] = node(2, 13, "retrieval_check", "build_index.py:49", "build", badges=(9,))
    nd["ss"] = node(3, 13, "similarity_search", "vectorstores.py:730", "lib")
    nd["sws"] = node(4, 13, "similarity_search\n_with_score", "vectorstores.py:817", "lib")
    nd["eq"] = node(5, 13, "embed_query", "embeddings.py:76", "embed", badges=(10,))

    # shared get_embedding: tall box at the right
    gx, gw, gy0, gy1 = C[6] - 10.5, 21.0, 8.0, 51.0
    box(ax, gx, gy0, gw, gy1 - gy0, fc=GREEN_L, ec=GREEN, lw=2.2)
    text(ax, gx + gw / 2, gy1 - 2.2, "get_embedding", size=10.5, family=MONO, weight="bold", ha="center")
    text(ax, gx + gw / 2, gy1 - 4.6, "embeddings.py:39-64", size=8, family=MONO, ha="center", color=GREEN)
    badge(ax, gx + 1.8, gy1 + 0.4, 6, r=1.35, size=9)
    box(ax, gx + 1.2, 22.0, gw - 2.4, 15.0, fc="white", ec=GRAY, lw=1.5, ls=(0, (4, 2)))
    text(ax, gx + gw / 2, 33.6, "native code", size=8.8, ha="center", weight="bold", color=GRAY)
    text(ax, gx + gw / 2, 30.6, "tokenizer", size=8.6, ha="center", color=GRAY)
    text(ax, gx + gw / 2, 28.0, "model forward", size=8.6, ha="center", color=GRAY)
    text(ax, gx + gw / 2, 25.0, "do NOT step in", size=8.2, ha="center", color=ORANGE, weight="bold")
    text(ax, gx + gw / 2, 42.0, "mode = 'sentence'", size=8.6, family=MONO, ha="center", weight="bold")
    text(ax, gx + gw / 2, 39.4, ":63  [CLS] vector", size=8.4, family=MONO, ha="center")
    text(ax, gx + gw / 2, 19.0, "mode = 'query'", size=8.6, family=MONO, ha="center", weight="bold")
    text(ax, gx + gw / 2, 16.4, ":59-60  mean pooling", size=8.4, family=MONO, ha="center")
    text(ax, gx + gw / 2, 11.6, "(Stop 6: sentence)", size=8.2, ha="center", color=MUTED)
    text(ax, gx + gw / 2, 9.4, "(Stop 10: query)", size=8.2, ha="center", color=MUTED)

    def link(a, b, rad=0.0, color=INK, lw=1.9, ls="-", frm="r", to="l"):
        xa = nd[a][1] if frm == "r" else (nd[a][0] if frm == "l" else (nd[a][0] + nd[a][1]) / 2)
        xb = nd[b][0] if to == "l" else (nd[b][1] if to == "r" else (nd[b][0] + nd[b][1]) / 2)
        ya = nd[a][2] + (NH / 2 if frm == "b" else (-NH / 2 if frm == "t" else 0))
        yb = nd[b][2] + (NH / 2 if to == "b" else (-NH / 2 if to == "t" else 0))
        arrow(ax, (xa, ya), (xb, yb), color=color, lw=lw, rad=rad, ls=ls)

    link("module", "main")
    for child in ("load", "init", "fd", "re", "rc"):
        link("main", child, rad=0.12 if nd[child][2] < 45 else -0.12)
    link("load", "clean"); link("init", "pre"); link("fd", "ft"); link("ft", "at"); link("at", "ed")
    link("rc", "ss"); link("ss", "sws"); link("sws", "eq")
    link("re", "nat", color=GRAY)
    arrow(ax, (nd["at"][0] + 3, nd["at"][2] - NH / 2), (nd["nat"][1] - 2, nd["nat"][2] + NH / 2), color=GRAY, rad=0.15, ls=(0, (4, 2)))
    arrow(ax, (nd["sws"][0] + 3, nd["sws"][2] + NH / 2), (nd["nat"][1] - 2, nd["nat"][2] - NH / 2), color=GRAY, rad=-0.15, ls=(0, (4, 2)))
    arrow(ax, (nd["ed"][1], 45), (gx, 45), color=GREEN, lw=2.2)
    arrow(ax, (nd["eq"][1], 13), (gx, 13), color=GREEN, lw=2.2)

    # legend
    ly = 3.0
    items = ((BLUE, BLUE_L, "-", "build_index.py"), (GREEN, GREEN_L, "-", "embeddings.py"), (GRAY, GRAY_L, "-", "library code (step into: justMyCode false)"),
             (GRAY, "white", (0, (4, 2)), "native code (cannot step in)"))
    lx = 4.0
    for c, l, ls, label in items:
        box(ax, lx, ly - 1.0, 2.6, 2.2, fc=l, ec=c, lw=1.6, ls=ls)
        text(ax, lx + 3.8, ly, label, size=9.3)
        lx += 3.8 + 0.62 * len(label) + 4.5
    save(fig, "03_call_map.png")


# ------------------------------------------------------------------ 04 stop map
def stop_map():
    fig, ax = canvas("The 11 stops on one page",
                     "Where to stop, what to watch, what you should see. Keep this open next to the debugger.")
    cols = {"n": 5.0, "where": 9.5, "what": 42.0, "watch": 63.0, "expect": 104.0, "tag": 143.5}
    top = 77.0
    box(ax, 3, top - 4.6, 154, 4.6, fc=INK, ec=INK)
    for key, label in (("where", "WHERE (file:line)"), ("what", "WHAT YOU SEE"), ("watch", "WATCH THESE"), ("expect", "EXPECT"), ("tag", "SOURCE")):
        text(ax, cols[key], top - 2.3, label, size=9.2, color="white", weight="bold")
    rows = [
        (1, "build_index.py:83", BLUE, "Guard + globals", "Locals: args  |  Globals: SERVER_DIR, REPO_ROOT\nsys.path[0]  |  Console: args.out_dir.exists()", "args.rebuild True; exists() False on 1st run", "real run"),
        (2, "build_index.py:37\n  if value not str/int/float/bool\nbuild_index.py:45  if i == 0", BLUE, "Metadata cleaning", "Locals: k, v, doc.metadata (before / after)", "languages: list -> '[\"eng\"]'", "estimate"),
        (3, "build_index.py:92", BLUE, "Loaded chunks", "Locals: chunks, lengths\nConsole: sum(1 for l in lengths if l < 50)", "753 chunks; 206 under 50 chars", "real run"),
        (4, "build_index.py:96 -> F11\nembeddings.py:37", GREEN, "The embedder", "Call Stack: __init__ <- main <- <module>\nLocals: self.device, batch_size, model.config", "cpu, 32; model.training False", "estimate"),
        (5, "embeddings.py:71 and :72\n  if start in (0, 736)", GREEN, "Chroma calls embedder", "Call Stack: 6 frames (see 03_call_map.png)\nLocals: texts, start, out", "len(texts) 753 (estimate)\n24 passes: 23 x 32 + 17", "arithmetic"),
        (6, "embeddings.py:55\n  if mode == \"sentence\"", GREEN, "One batch to vectors", "Console: shapes of inp, output.last_hidden_state\nF10 to :63 -> vectors.shape; do not step into :55", "(32,L) L<=512 -> (32,L,hidden) -> (32,hidden)", "estimate"),
        (7, "build_index.py:104", BLUE, "Index exists", "Console: _collection.count(), .metadata,\n.configuration, list(args.out_dir.iterdir())", "753; l2 (estimate); sqlite + UUID folder", "real run"),
        (8, "build_index.py:111", BLUE, "Reopen from disk", "Locals: count, len(chunks), size_mb", "753; about 6.3 MB", "real run"),
        (9, "build_index.py:58  then F10", BLUE, "The sample", "Console: (len(sample)-1)*step\nsample[-1]['chunk_id'] vs questions[-1]['chunk_id']", "1434 -> 478 -> 400; last index 1197;\nthe last 16% is never tested", "real run"),
        (10, "embeddings.py:80\nbuild_index.py:62  hit count 1", GREEN, "One real search", "Call Stack: 6 frames  |  Locals: mode == \"query\"\nConsole: similarity_search_with_score(questions[700]..)", "chunk 364 (~16.3), then 362 (~22.0)", "real run"),
        (11, "build_index.py:117", BLUE, "The verdict", "Locals: res", "recall@2 0.785, @5 0.88;\ncentre-chunk 0.34 and 0.465", "real run"),
    ]
    RH = 5.95
    y = top - 4.6
    for i, (n, where, col, what, watch, expect, tg) in enumerate(rows):
        yb = y - RH
        ax.add_patch(Rectangle((3, yb), 154, RH, fc=(GRAY_L if i % 2 == 0 else "white"), ec="none", zorder=1))
        ax.add_patch(Rectangle((3, yb), 0.9, RH, fc=col, ec="none", zorder=2))
        badge(ax, cols["n"] + 0.6, yb + RH / 2, n)
        text(ax, cols["where"], yb + RH / 2, where, size=8.2, family=MONO, color=INK)
        text(ax, cols["what"], yb + RH / 2, what, size=9.6, weight="bold")
        text(ax, cols["watch"], yb + RH / 2, watch, size=8.6)
        text(ax, cols["expect"], yb + RH / 2, expect, size=8.6)
        text(ax, cols["tag"], yb + RH / 2, tg, size=8.6, family=MONO, color=TAG_COLOR[tg], weight="bold")
        y = yb
    # legend
    ly = 1.6
    for k, (c, label) in enumerate(((BLUE, "build_index.py"), (GREEN, "embeddings.py"))):
        ax.add_patch(Rectangle((4 + 30 * k, ly - 0.8), 1.6, 1.6, fc=c, ec="none"))
        text(ax, 6.4 + 30 * k, ly, label, size=9)
    text(ax, 70, ly, "Values come from: real run (an earlier run), arithmetic (we calculate it), estimate (a guess from the code).", size=9, color=MUTED)
    save(fig, "04_stop_map.png")


if __name__ == "__main__":
    big_picture()
    two_paths()
    call_map()
    stop_map()
