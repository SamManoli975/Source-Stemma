# Stemmatology for AI Source Attribution — Project Plan

## Elevator pitch

Every LLM answer is built out of sources — retrieved documents, training data echoes, or both. Right now that lineage is either invisible or flattened into a list of footnotes. This project treats an LLM's source material the way classical philology treats manuscripts: as witnesses descended from (and corrupted/paraphrased/conflated relative to) one another. Instead of a flat citation list, the user gets a **stemma** — a family tree of sources showing which one is most authentic/original, how the others relate to it (copy, paraphrase, conflation, independent parallel), and where a newly-supplied document would slot in if dropped into the tree.

Interaction model: **Connected Papers, but the edges are philological relationships instead of citation-graph similarity, and the tree updates live as you drag new sources in.**

This builds directly on the pairwise/localize/relate/evidence pipeline already in progress (`~/pairwise/`) — this document is about turning that pipeline into a demonstrable end-to-end product with a visual front end, and about the additional relation-detection and layout logic needed to make the "family tree" metaphor actually work.

---

## 1. Core pipeline (three stages)

### Stage A — Provenance-preserving retrieval
When the LLM answers a query, capture *every* source it drew on with full provenance intact (not just a URL — the actual text span, retrieval rank, timestamp/version if available). This is the "retrieval provenance" half already scoped in the existing RAG work (`chunk.py` → `index.py` → `retrieve.py` → `generate.py`). Nothing novel here methodologically; it's the input layer that feeds Stage B.

### Stage B — Textual criticism & relation detection
For every pair of sources (and the generated output itself, treated as a witness), run the collation/variant-analysis machinery already built (`pairwise.py`, `localize.py`) to classify the relationship. This is the heart of the novel contribution. Candidate relation types to detect (see §2 below) go here, plus:
- **Date/version analysis** — where dates or version metadata exist, use them as a prior on directionality (older is a more plausible ancestor, but not proof — contamination and back-dating both happen with digital sources too). Where no dates exist, fall back to purely text-internal criteria (Lachmann-style: shared error = descent signal, shared correct reading = no signal), which is already the pipeline's design.

### Stage C — Tree construction & authenticity scoring
Convert the pairwise relation/distance matrix into a tree (or DAG, if contamination/conflation is detected — classical stemmatics increasingly accepts trees aren't always trees, and a phylogenetic *network* is often more honest than forcing bifurcation). Score each node for "authenticity" — proximity to the reconstructed archetype / lowest cumulative variant distance — and surface that as the headline number the user sees ("most authentic source").

---

## 2. Relation-detection taxonomy

This is the feature likely to differentiate the project — Connected Papers only has one edge type (citation similarity). A philologically-grounded tool can distinguish:

| Relation | Signal | Existing/planned code |
|---|---|---|
| Verbatim copy | Near-identical span, Smith–Waterman alignment score ≈ 1 | `localize.py` |
| Paraphrase | High semantic similarity, low lexical overlap | `localize.py` (sentence-transformers tier) |
| Quotation vs. commentary | Asymmetric length + framing markers ("as X says...") | `relate.py` |
| Conjunctive error / shared corruption | Shared *non-obvious* deviation from a reconstructed reading | `pairwise.py` (rootless character-matrix approach) |
| Independent parallel (polygenesis) | Similar content, no shared error, plausible common knowledge | Needs explicit negative-control style test — see open issues |
| Contamination / conflation | A witness shares errors with two otherwise-unrelated branches | Requires DAG, not tree — on the horizon |
| Interpolation | A span present in one witness with no plausible ancestor | Localize + evidence.py distinctive-phrase extraction |

Each of these becomes a distinct edge type/color in the visualization, which is also what makes the tree readable rather than a spaghetti graph.

---

## 3. The "Connected Papers" interaction layer

What to borrow from Connected Papers' UX, and what to change:

- **Force-directed / radial layout** with the candidate archetype (or the LLM output itself) at the center, and witnesses arranged by distance-to-center = authenticity/fidelity, not by publication date.
- **Node size** = corpus weight or citation frequency (optional secondary signal); **edge thickness** = strength of shared-variant evidence; **edge color** = relation type from the taxonomy above.
- **Drag-and-drop insertion**: user drops a new article/snippet onto the canvas. Pipeline runs Stage B against every existing node (or a pruned candidate set via embedding pre-filter, for speed), computes its best-fit position, and animates it sliding into place on the tree — this is the single most demo-able feature and worth prioritizing early even with a rough version.
- **Click a node** → side panel shows the actual variant evidence (the specific shared phrases / errors that justified the edge), not just a similarity score — this is what makes it feel *forensic* rather than a black-box embedding distance, consistent with the project's existing evidentiary/source-critical framing.
- **"Why is this the most authentic source?"** explainer panel — walks through the support statistics (bootstrap/Bremer-style, already on the roadmap) in plain language.

---

## 4. Relationship to the existing four-phase plan

This product/UI layer doesn't replace the existing research phases — it's the demonstrator that sits on top of them:

- **Phase 0** (engine-agnostic collation-to-tree pipeline, known ground truth) → this is what powers Stages B/C above; get the Polycarp corpus rendering as a static tree first, no UI polish.
- **Phase 1** (support statistics + negative control) → this is what makes the "most authentic" score and the drag-and-drop placement *trustworthy* rather than just a nice picture — don't ship the confident-looking UI before this lands, or it'll oversell an uncalibrated pipeline.
- **Phase 2** (DeepSeek demonstration) → this is where "AI source attribution" stops being a metaphor and becomes literal — tracing a live model's generated span back through retrieved sources using the same tree.
- **Phase 3** (writeup + provider recommendation) → the interactive tool becomes the artifact you show alongside the paper, not a replacement for it.

Suggested build order for the FYP itself: get static tree generation solid on Polycarp data (Phase 0/1 territory) → wrap it in a minimal web UI (even a simple D3/force-graph render) → add drag-and-drop once the underlying "place a new node" function exists as a callable pipeline step → only then invest in the fuller Connected-Papers-style polish (animations, side panels, relation-type legends).

---

## 5. Open problems worth flagging in the plan

- **Directionality without dates**: for digital/AI-generated sources, "older" doesn't map cleanly onto "more original" the way it does with manuscripts — an LLM can generate text that looks archetypal by coincidence. The date-based prior needs to be explicitly weighted *below* the text-internal (Lachmannian) signal, not above it.
- **Trees vs. networks**: once contamination/conflation detection is in scope, forcing a strict bifurcating tree will misrepresent the data. Worth deciding early whether the visualization commits to a tree (simpler, more legible) or a DAG (more honest, harder to lay out and to explain to a user).
- **Translation layer**: already flagged in the existing project notes — English-text NLP tools can't recover relationships that only exist at the level of the original-language manuscript tradition. For the AI-attribution use case this is less severe (LLM outputs and their sources are usually all in the working language), but worth stating explicitly as a scope boundary.
- **Speed at insertion time**: full pairwise comparison against every existing node on every drag-and-drop is O(n) collation calls, which is fine for a demo corpus and potentially slow for a large one — an embedding-based pre-filter to shortlist candidates before running the expensive collation step is worth designing in from the start rather than retrofitting.

---

## 6. Feature ideas (not yet scoped into phases — parking lot)

- Confidence-shaded edges (opacity = statistical support) rather than binary present/absent relations.
- Timeline scrubber, for corpora where dates *are* reliable, to watch the tree's shape change as sources are added chronologically.
- Export the tree + evidence as a citable appendix (ties back into the "recommendation to AI providers" final goal — a provider could audit a specific output's tree).
- A "negative control" toggle in the UI itself — show the tool failing gracefully/visibly on unrelated documents, which doubles as a built-in credibility demonstration rather than something buried in the methodology section.
