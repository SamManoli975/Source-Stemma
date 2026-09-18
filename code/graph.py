"""
graph.py — paste a list of URLs, get an interactive dependency graph.

Pipeline:
  1. Fetch each URL, extract main body text + publication date from metadata.
  2. Run relate() on every pair to detect which source depends on which.
  3. Render a self-contained HTML file (vis.js DAG) — open in any browser.

Usage:
  python3 graph.py https://... https://...        # args
  cat urls.txt | python3 graph.py                 # stdin, one URL per line
  python3 graph.py                                 # interactive: paste URLs, Ctrl-D to finish
"""

from __future__ import annotations

import sys
import json
import os
import itertools
from datetime import date, datetime
from html.parser import HTMLParser

try:
    import requests
    import certifi
    _USE_REQUESTS = True
except ImportError:
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError
    _USE_REQUESTS = False

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "text"))

from relate import relate
from evidence import extract_evidence
from citations import fingerprint, citation_dependency_score


# ══════════════════════════════════════════════════════════════════════════
# HTML SCRAPER
# ══════════════════════════════════════════════════════════════════════════

_SKIP_TAGS = {"script", "style", "nav", "header", "footer",
              "aside", "noscript", "form", "button", "svg"}
_BODY_TAGS  = {"article", "main", "section", "p", "h1", "h2", "h3",
               "h4", "h5", "h6", "li", "td", "th", "blockquote", "pre"}


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title        = ""
        self.date         = None      # datetime.date or None
        self._in_title    = False
        self._skip_depth  = 0
        self._chunks: list[str] = []

    # ---- tag events -------------------------------------------------------

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)

        # title element
        if tag == "title":
            self._in_title = True
            return

        # open-graph / meta date fields
        if tag == "meta":
            prop    = attrs_d.get("property", "") or attrs_d.get("name", "")
            content = attrs_d.get("content", "")
            if prop.lower() in (
                "article:published_time", "og:article:published_time",
                "pubdate", "date", "article:modified_time",
            ):
                self.date = self.date or _parse_date(content)

        # <time datetime="...">
        if tag == "time" and not self.date:
            dt_attr = attrs_d.get("datetime", "")
            if dt_attr:
                self.date = _parse_date(dt_attr)

        # skip noisy sections
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    # ---- accessor ---------------------------------------------------------

    def body_text(self) -> str:
        return " ".join(self._chunks)


def _parse_date(s: str) -> date | None:
    """Try common date formats. Returns None on failure."""
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt).date()
        except (ValueError, TypeError):
            pass
    return None


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; PairwiseBot/1.0; "
        "+https://github.com/pairwise)"
    )
}


def fetch_page(url: str) -> dict:
    """Fetch a URL, extract body text and publication date.
    Returns { url, title, text, date } where date may be None."""
    try:
        if _USE_REQUESTS:
            resp = requests.get(url, headers=_HEADERS, timeout=15,
                                verify=certifi.where())
            resp.raise_for_status()
            html = resp.text
        else:
            from urllib.request import urlopen, Request
            from urllib.error import URLError, HTTPError
            req = Request(url, headers=_HEADERS)
            try:
                with urlopen(req, timeout=15) as r:
                    raw = r.read()
                    charset = r.headers.get_content_charset() or "utf-8"
                    try:
                        html = raw.decode(charset, errors="replace")
                    except LookupError:
                        html = raw.decode("utf-8", errors="replace")
            except (URLError, HTTPError) as e:
                print(f"  [WARN] Could not fetch {url}: {e}", file=sys.stderr)
                return {"url": url, "title": url, "text": "", "date": None}
    except Exception as e:
        print(f"  [WARN] Could not fetch {url}: {e}", file=sys.stderr)
        return {"url": url, "title": url, "text": "", "date": None}

    parser = _PageParser()
    parser.feed(html)

    title = parser.title.strip() or url
    text  = parser.body_text()

    return {"url": url, "title": title, "text": text, "date": parser.date}


def fetch_all(urls: list[str]) -> list[dict]:
    nodes = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] fetching {url} …")
        node = fetch_page(url)
        node["id"] = i - 1
        nodes.append(node)
        chars = len(node["text"])
        date_s = str(node["date"]) if node["date"] else "date unknown"
        print(f"       → {chars} chars  |  {date_s}  |  {node['title'][:60]}")
    return nodes


# ══════════════════════════════════════════════════════════════════════════
# PAIRWISE RELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def build_edges(nodes: list[dict]) -> list[dict]:
    """Detect dependency edges using three complementary signals:

    1. CITATION FINGERPRINTS — footnotes, inline citations (Author, Year),
       bibliography entries, author-name mentions with cite-phrase verbs,
       title-keyword hits.  Highest confidence; directional by definition.

    2. RELATE() — verbatim / paraphrase overlap + coverage-based role
       classification.  Good for catching quotations.

    3. DATE ORDER — if both signals are ambiguous, fall back to date order.

    An edge is drawn when the combined citation score is ≥ 0.2 OR relate()
    finds a non-trivial overlap.  Edges where the only signal is shared
    vocabulary with no citation structure are suppressed.
    """
    # build fingerprints for every node up front
    fps = {nd["id"]: fingerprint(nd, [o for o in nodes if o["id"] != nd["id"]])
           for nd in nodes}
    for nd in nodes:
        print(f"  fingerprint [{nd['id']}] {nd['title'][:50]}")
        print(f"    {fps[nd['id']].report().splitlines()[1]}")
        print(f"    {fps[nd['id']].report().splitlines()[2]}")

    edges = []
    pairs = list(itertools.combinations(range(len(nodes)), 2))
    total = len(pairs)

    for idx, (i, j) in enumerate(pairs, 1):
        a, b = nodes[i], nodes[j]
        if not a["text"] or not b["text"]:
            continue
        print(f"\n  pair {idx}/{total}: [{i}] vs [{j}]")

        fp_a, fp_b = fps[a["id"]], fps[b["id"]]

        # ── citation scores in both directions ──────────────────────────────
        # score_ab: how strongly does B cite/depend on A?
        # score_ba: how strongly does A cite/depend on B?
        score_ab, ev_ab = citation_dependency_score(fp_a, fp_b, a, b)
        score_ba, ev_ba = citation_dependency_score(fp_b, fp_a, b, a)

        print(f"    citation A→B: {score_ab:.2f}  {ev_ab}")
        print(f"    citation B→A: {score_ba:.2f}  {ev_ba}")

        # ── semantic relate() ───────────────────────────────────────────────
        rel = relate(a["text"], b["text"],
                     date_a=a["date"], date_b=b["date"])
        print(f"    relate(): {rel.overlap.verdict} score={rel.overlap.score:.2f}"
              f"  prior={rel.prior}  basis={rel.priority_basis}")

        # ── decide direction and whether to draw an edge ────────────────────
        CITE_THRESHOLD = 0.2   # minimum citation score to trust
        SEMANTIC_THRESHOLD = 0.6  # only trust relate() when score is strong

        cite_signal  = max(score_ab, score_ba) >= CITE_THRESHOLD
        semantic_ok  = (rel.overlap.verdict != "no_overlap"
                        and rel.overlap.score >= SEMANTIC_THRESHOLD
                        and rel.prior != "unknown")

        if not cite_signal and not semantic_ok:
            # shared vocabulary only — siblings or unrelated
            print(f"    → suppressed (no citation signal, weak semantic overlap)")
            continue

        # choose direction: citation beats semantic beats date
        if cite_signal and score_ab != score_ba:
            # clear citation direction
            if score_ab > score_ba:
                src_id, tgt_id = a["id"], b["id"]
                basis = "citation"
                evidence_strs = ev_ab
                score = score_ab
            else:
                src_id, tgt_id = b["id"], a["id"]
                basis = "citation"
                evidence_strs = ev_ba
                score = score_ba
        elif semantic_ok and rel.prior != "unknown":
            src_id = a["id"] if rel.prior == "A" else b["id"]
            tgt_id = b["id"] if rel.prior == "A" else a["id"]
            basis = rel.priority_basis
            evidence_strs = []
            score = rel.overlap.score
        elif a.get("date") and b.get("date") and a["date"] != b["date"]:
            # fall back to date order
            if a["date"] < b["date"]:
                src_id, tgt_id = a["id"], b["id"]
            else:
                src_id, tgt_id = b["id"], a["id"]
            basis = "date"
            evidence_strs = []
            score = max(score_ab, score_ba)
        else:
            print(f"    → suppressed (ambiguous direction)")
            continue

        # ── build snippet for tooltip ───────────────────────────────────────
        snippet_parts = []
        if evidence_strs:
            snippet_parts.append(" · ".join(evidence_strs[:3]))
        if rel.overlap.verdict in ("verbatim", "paraphrase"):
            ev = extract_evidence(a["text"], b["text"])
            if ev.shared_phrases:
                snippet_parts.append(f'"{ev.shared_phrases[0].text[:100]}"')
        snippet = " | ".join(snippet_parts)[:200]

        verdict = "citation" if basis == "citation" else rel.overlap.verdict

        edges.append({
            "from":     src_id,
            "to":       tgt_id,
            "verdict":  verdict,
            "score":    round(score, 3),
            "snippet":  snippet,
            "basis":    basis,
            "role_src": rel.role_a if rel.prior == "A" else rel.role_b,
            "role_tgt": rel.role_b if rel.prior == "A" else rel.role_a,
        })
        print(f"    → EDGE {src_id}→{tgt_id}  verdict={verdict}  score={score:.2f}"
              f"  basis={basis}")

    return edges


# ══════════════════════════════════════════════════════════════════════════
# VIS.JS HTML RENDERER
# ══════════════════════════════════════════════════════════════════════════

def _date_sort_key(node: dict) -> int:
    """Return an integer year for vertical positioning (earlier = higher).
    Unknown dates get year 9999 (bottom)."""
    d = node.get("date")
    if d is None:
        return 9999
    return d.year


_VIS_CDN = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"
_VIS_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          ".vis-network.min.js")


def _vis_script_tag() -> str:
    """Return a <script> tag with vis.js inlined. Downloads once, caches."""
    if os.path.exists(_VIS_CACHE):
        with open(_VIS_CACHE, encoding="utf-8") as f:
            js = f.read()
    else:
        print("  downloading vis.js (one-time, ~640 KB) …")
        try:
            if _USE_REQUESTS:
                r = requests.get(_VIS_CDN, verify=certifi.where(), timeout=30)
                r.raise_for_status()
                js = r.text
            else:
                from urllib.request import urlopen
                with urlopen(_VIS_CDN, timeout=30) as r:
                    js = r.read().decode("utf-8")
            with open(_VIS_CACHE, "w", encoding="utf-8") as f:
                f.write(js)
        except Exception as e:
            print(f"  [WARN] Could not download vis.js: {e}. "
                  f"Graph will need internet connection to render.")
            return (f'<script src="{_VIS_CDN}"></script>')
    return f"<script>\n{js}\n</script>"


def _compute_positions(nodes: list[dict], edges: list[dict]) -> dict[int, tuple[float, float]]:
    """Compute 2D (x, y) for each node.

    X axis = time: nodes are spread left-to-right by date rank.
             Undated nodes are spread evenly; dated ones honour their order.
    Y axis = dependency depth: sources (no incoming edges) sit at the top;
             each node is placed one step below its deepest ancestor.

    Returns {node_id: (x, y)} in canvas units (pixels).
    """
    n = len(nodes)
    if n == 0:
        return {}

    # --- X: rank by date ---
    sorted_by_date = sorted(nodes, key=_date_sort_key)
    x_rank = {nd["id"]: i for i, nd in enumerate(sorted_by_date)}

    # --- Y: topological depth (longest incoming path) ---
    depth: dict[int, int] = {nd["id"]: 0 for nd in nodes}
    # iterate until stable (handles cycles by capping at n iterations)
    for _ in range(n):
        changed = False
        for e in edges:
            src, tgt = e["from"], e["to"]
            if depth[src] + 1 > depth[tgt]:
                depth[tgt] = depth[src] + 1
                changed = True
        if not changed:
            break

    max_depth = max(depth.values()) if depth else 0
    # spread so nodes are readable
    x_spacing = max(220, 900 // max(1, n - 1))
    y_spacing = 180

    pos = {}
    for nd in nodes:
        nid = nd["id"]
        x   = x_rank[nid] * x_spacing
        y   = depth[nid]  * y_spacing
        pos[nid] = (x, y)

    # centre the whole graph around (0, 0)
    xs = [v[0] for v in pos.values()]
    ys = [v[1] for v in pos.values()]
    cx, cy = sum(xs) / n, sum(ys) / n
    return {nid: (x - cx, y - cy) for nid, (x, y) in pos.items()}


def render_html(nodes: list[dict], edges: list[dict]) -> str:
    pos = _compute_positions(nodes, edges)

    vis_nodes = []
    for node in nodes:
        date_str  = str(node["date"]) if node["date"] else "date unknown"
        label     = _short_label(node["title"])
        color     = "#4A90D9" if node["date"] else "#999999"
        tooltip   = (f"<b>{node['title']}</b><br>"
                     f"Date: {date_str}<br>"
                     f"<a href='{node['url']}' target='_blank'>{node['url'][:60]}</a>")
        x, y = pos.get(node["id"], (0, 0))
        vis_nodes.append({
            "id":    node["id"],
            "label": label,
            "title": tooltip,
            "x":     round(x),
            "y":     round(y),
            # no "fixed" — nodes are fully draggable
            "color": {"background": color, "border": "#2c5f8a",
                      "highlight": {"background": "#6db3f2"}},
            "font":  {"color": "#ffffff", "size": 13},
            "shape": "box",
            "margin": 10,
            "url":   node["url"],
        })

    vis_edges = []
    for i, e in enumerate(edges):
        color = "#3498db" if e["verdict"] == "citation" else \
                "#2ecc71" if e["verdict"] == "verbatim" else "#e67e22"
        label = f"{e['verdict']}\n{e['score']:.2f}"
        tooltip = (f"<b>Dependency: {e['verdict']}</b><br>"
                   f"Score: {e['score']:.2f}<br>"
                   f"Basis: {e['basis']}<br>"
                   f"Roles: {e['role_src']} → {e['role_tgt']}<br>"
                   f"<i>{e['snippet']}</i>")
        vis_edges.append({
            "id":     i,
            "from":   e["from"],
            "to":     e["to"],
            "label":  label,
            "title":  tooltip,
            "arrows": "to",
            "color":  {"color": color, "highlight": color},
            "font":   {"size": 11, "align": "middle"},
            "width":  2,
        })

    nodes_json = json.dumps(vis_nodes, ensure_ascii=False, indent=2)
    edges_json = json.dumps(vis_edges, ensure_ascii=False, indent=2)
    vis_tag    = _vis_script_tag()

    return _HTML_TEMPLATE.replace("__VIS_SCRIPT__", vis_tag)\
                          .replace("__NODES__", nodes_json)\
                          .replace("__EDGES__", edges_json)


def _short_label(title: str, max_len: int = 30) -> str:
    """Wrap title for a vis.js box node label (newline at word boundary)."""
    words = title.split()
    lines, line = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > max_len:
            if len(line) > 1:
                lines.append(" ".join(line[:-1]))
                line = [line[-1]]
            else:
                lines.append(" ".join(line))
                line = []
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines[:3])  # cap at 3 lines


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pairwise Dependency Graph</title>
__VIS_SCRIPT__
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }
  #toolbar {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px; background: #16213e; border-bottom: 1px solid #0f3460;
  }
  #toolbar h1 { font-size: 16px; font-weight: 600; color: #4A90D9; }
  #toolbar .legend { display: flex; gap: 16px; font-size: 12px; }
  .legend-item { display: flex; align-items: center; gap: 5px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; }
  .dot-citation  { background: #3498db; }
  .dot-verbatim  { background: #2ecc71; }
  .dot-paraphrase{ background: #e67e22; }
  .dot-dated     { background: #4A90D9; }
  .dot-undated   { background: #999; }
  #freeze-btn {
    margin-left: auto; padding: 4px 12px; background: #0f3460;
    border: 1px solid #4A90D9; border-radius: 4px; color: #eee;
    cursor: pointer; font-size: 12px;
  }
  #freeze-btn:hover { background: #4A90D9; color: #fff; }
  #network { width: 100%; height: calc(100vh - 52px); }
  #tooltip-box {
    position: absolute; background: #0f3460; border: 1px solid #4A90D9;
    border-radius: 6px; padding: 8px 12px; font-size: 12px; max-width: 320px;
    pointer-events: none; display: none; z-index: 10; line-height: 1.5;
  }
</style>
</head>
<body>
<div id="toolbar">
  <h1>Pairwise Dependency Graph</h1>
  <div class="legend">
    <div class="legend-item"><div class="dot dot-citation"></div>citation (explicit)</div>
    <div class="legend-item"><div class="dot dot-verbatim"></div>verbatim dependency</div>
    <div class="legend-item"><div class="dot dot-paraphrase"></div>paraphrase dependency</div>
    <div class="legend-item"><div class="dot dot-dated"></div>dated source</div>
    <div class="legend-item"><div class="dot dot-undated"></div>date unknown</div>
  </div>
  <button id="freeze-btn" onclick="togglePhysics()">⏸ Freeze</button>
</div>
<div id="network"></div>
<div id="tooltip-box"></div>

<script>
const nodesData = __NODES__;
const edgesData = __EDGES__;

const container = document.getElementById("network");
const data = {
  nodes: new vis.DataSet(nodesData),
  edges: new vis.DataSet(edgesData),
};

const options = {
  layout: { randomSeed: 0 },
  physics: {
    enabled: true,
    solver: "forceAtlas2Based",
    forceAtlas2Based: {
      gravitationalConstant: -80,
      centralGravity: 0.005,
      springLength: 220,
      springConstant: 0.10,
      damping: 0.6,
      avoidOverlap: 0.8,
    },
    stabilization: {
      enabled: true,
      iterations: 300,
      updateInterval: 10,
      fit: true,
    },
    minVelocity: 0.5,
  },
  interaction: {
    hover: true,
    tooltipDelay: 150,
    navigationButtons: true,
    keyboard: true,
    zoomView: true,
    dragView: true,
    dragNodes: true,
  },
  edges: {
    smooth: false,
    arrowStrikethrough: false,
  },
  nodes: { borderWidth: 1, shadow: { enabled: true, size: 8, x: 3, y: 3 } },
};

const network = new vis.Network(container, data, options);

// clicking a node opens its URL
network.on("click", function(params) {
  if (params.nodes.length === 1) {
    const node = data.nodes.get(params.nodes[0]);
    if (node && node.url) window.open(node.url, "_blank");
  }
});

// after dragging a node, briefly re-enable physics so edges settle, then stop
network.on("dragEnd", function(params) {
  if (params.nodes.length > 0 && !physicsOn) return;
  if (params.nodes.length > 0) {
    network.setOptions({ physics: { enabled: true } });
    setTimeout(() => { if (!physicsOn) network.setOptions({ physics: { enabled: false } }); }, 1500);
  }
});

// freeze / unfreeze button
let physicsOn = true;
function togglePhysics() {
  physicsOn = !physicsOn;
  network.setOptions({ physics: { enabled: physicsOn } });
  document.getElementById("freeze-btn").textContent = physicsOn ? "⏸ Freeze" : "▶ Unfreeze";
}

// auto-freeze once stabilisation finishes
network.on("stabilizationIterationsDone", function() {
  physicsOn = false;
  network.setOptions({ physics: { enabled: false } });
  network.fit({ animation: { duration: 600, easingFunction: "easeInOutQuad" } });
  document.getElementById("freeze-btn").textContent = "▶ Unfreeze";
});

// richer HTML tooltip for edges (vis.js uses the title field)
const tip = document.getElementById("tooltip-box");
network.on("hoverEdge", function(params) {
  const edge = data.edges.get(params.edge);
  if (!edge || !edge.title) return;
  tip.innerHTML = edge.title;
  tip.style.display = "block";
});
network.on("blurEdge", function() { tip.style.display = "none"; });
document.addEventListener("mousemove", function(e) {
  tip.style.left = (e.clientX + 14) + "px";
  tip.style.top  = (e.clientY + 10) + "px";
});
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def _read_urls() -> list[str]:
    """Collect URLs from sys.argv or stdin."""
    if len(sys.argv) > 1:
        return [u.strip() for u in sys.argv[1:] if u.strip()]

    print("Paste URLs (one per line). Press Ctrl-D when done:")
    urls = []
    try:
        for line in sys.stdin:
            line = line.strip()
            if line:
                urls.append(line)
    except KeyboardInterrupt:
        pass
    return urls


def main():
    urls = _read_urls()
    if not urls:
        print("No URLs provided. Exiting.")
        sys.exit(1)

    print(f"\n── Fetching {len(urls)} page(s) ──────────────────────────────")
    nodes = fetch_all(urls)

    active = [n for n in nodes if n["text"]]
    if len(active) < 2:
        print("\n[WARN] Fewer than 2 pages had extractable text. "
              "No edges can be computed.")

    print(f"\n── Comparing {len(active)} source(s) ({len(list(itertools.combinations(active, 2)))} pairs) ──")
    edges = build_edges(nodes)

    print(f"\n── Rendering HTML ({len(nodes)} nodes, {len(edges)} edges) ──")
    html = render_html(nodes, edges)

    out_path = os.path.join(os.path.dirname(__file__), "dependency_graph.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nSaved: {out_path}")
    print("Open it in your browser to view the interactive dependency graph.")


if __name__ == "__main__":
    main()
