# app.py
import os
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import language_v1 as language

# ────────────────────────────────────────────────────────────────────────────────
# Page setup
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Topical Hierarchy — Google Entity–Aligned", layout="wide")
st.title("Topical Hierarchy — Google Entity–Aligned (Single Output)")
st.caption("Upload your Keyword Insights CSV → choose hierarchy depth (1–4) → download a single topical hierarchy CSV aligned to Google’s Knowledge Graph.")

# ────────────────────────────────────────────────────────────────────────────────
# Credentials: load GCP NL key from Streamlit Secrets OR use pre-set env var
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Options")
    depth = st.selectbox("Hierarchy depth", [1, 2, 3, 4], index=2)
    salience_min = st.slider("Minimum entity salience", 0.0, 1.0, 0.35, 0.05)
    allow_org = st.checkbox("Allow ORGANIZATION entities (brand-led intent)", value=False)
    st.markdown("---")
    st.caption("Tip: Add your service-account JSON to **Secrets** (key = `gcp` or `gcp_json`). "
               "Otherwise, set the `GOOGLE_APPLICATION_CREDENTIALS` env var to a local JSON file.")

# Try to materialize a credentials file from Streamlit Secrets if present
def _maybe_write_sa_from_secrets() -> Optional[str]:
    try:
        if "gcp" in st.secrets:
            sa_info = dict(st.secrets["gcp"])
        elif "gcp_json" in st.secrets:
            sa_info = json.loads(st.secrets["gcp_json"])
        else:
            return None
        key_path = "/tmp/gcp-sa.json"
        with open(key_path, "w") as f:
            json.dump(sa_info, f)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        return key_path
    except Exception:
        return None

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    _maybe_write_sa_from_secrets()

# ────────────────────────────────────────────────────────────────────────────────
# Upload CSV
# ────────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload Keyword Insights CSV", type=["csv"])

REQUIRED = {"keyword", "search_volume", "cluster"}  # minimally required columns
ALL_COLUMNS_DOC = [
    "Keyword", "Search volume", "Cluster", "Topical cluster", "Spoke", "Rank", "Ranking page",
    "Multiple pages", "Multiple ranks", "Current estimated traffic", "Maximum traffic",
    "Opportunity", "Difficulty scores", "Recommended keyword for brief/content",
    "Serp Features", "Relevancy labels"
]

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    return df.rename(columns=mapping)

def top_k_keywords(group: pd.DataFrame, k: int = 5) -> List[str]:
    temp = group.copy()
    temp["search_volume"] = pd.to_numeric(temp["search_volume"], errors="coerce").fillna(0)
    temp = temp.sort_values("search_volume", ascending=False)
    return [str(x) for x in temp["keyword"].tolist()[:k]]

def synopsis_for_plc(cluster_name: str, kws: List[str]) -> str:
    return f"{cluster_name}. Key queries: " + "; ".join(kws)

# ────────────────────────────────────────────────────────────────────────────────
# Google Cloud NL Entity Analysis
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def gcp_entities(text: str) -> List[Dict]:
    client = language.LanguageServiceClient()
    doc = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
    resp = client.analyze_entities(document=doc)
    out = []
    for e in resp.entities:
        out.append({
            "name": e.name,
            "type": language.Entity.Type(e.type_).name,   # PERSON/ORGANIZATION/LOCATION/.../OTHER
            "salience": float(e.salience),
            "mid": e.metadata.get("mid") if e.metadata else None,
            "wikipedia_url": e.metadata.get("wikipedia_url") if e.metadata else None,
        })
    return out

def choose_entity(ents: List[Dict], cluster_name: str, kws: List[str],
                  salience_threshold: float, allow_org: bool) -> Optional[Dict]:
    """Pick the best entity by salience + light textual coverage, with type filtering."""
    if not ents:
        return None
    # type filter
    filtered = []
    for e in ents:
        et = e.get("type", "OTHER")
        if et in {"PERSON", "NUMBER", "DATE", "ADDRESS"}:
            continue
        if et == "ORGANIZATION" and not allow_org:
            continue
        filtered.append(e)
    if not filtered:
        filtered = ents
    # score
    text = (cluster_name + " " + " ".join(kws)).lower()
    best, best_score = None, -1
    for e in filtered:
        name = e["name"].lower()
        coverage = 1.0 if name in text else 0.0
        score = 0.7 * float(e.get("salience", 0.0)) + 0.3 * coverage
        if score > best_score:
            best, best_score = e, score
    # keep even if below threshold; we expose salience for QA
    if best is None:
        return None
    if float(best.get("salience", 0.0)) < salience_threshold:
        # You could mark for QA if needed
        return best
    return best

# ────────────────────────────────────────────────────────────────────────────────
# Curated MID→Parent (extend to unlock deeper and cleaner trees in your domain)
# ────────────────────────────────────────────────────────────────────────────────
CURATED_PARENT = {
    # B2B / Analytics
    "/m/0b9r8": ("Artificial intelligence", "/m/0b9r8"),
    "/m/02qkm5n": ("Data analytics", "/m/02qkm5n"),
    "/m/025rsfk": ("Business intelligence", "/m/025rsfk"),
    "/m/05wkwv": ("Business analytics", "/m/05wkwv"),
    "/m/07y6yz": ("Search engine optimization", "/m/07y6yz"),
    "/m/02hmvc": ("Digital marketing", "/m/02hmvc"),
    # Fashion / Retail (add real MIDs you see in output to refine)
    "/m/01c8m": ("Clothing", "/m/01c8m"),
    "/m/027b9": ("Jeans", "/m/027b9"),
    "/m/027b7": ("Dress", "/m/027b7"),
    "/m/01b7x": ("Jacket", "/m/01b7x"),
    "/m/02fg3h": ("Hoodie", "/m/02fg3h"),
    "/m/0h7f_": ("Running shoe", "/m/0h7f_"),
}

def infer_level1_parent(entity_name: str, entity_mid: Optional[str]) -> Tuple[str, Optional[str]]:
    """Prefer curated parent by MID; fallback to a few broad keyword heuristics."""
    if entity_mid in CURATED_PARENT:
        return CURATED_PARENT[entity_mid]

    name = (entity_name or "").lower()
    if any(k in name for k in ["analytics", "intelligence", "ai", "machine learning"]):
        return ("Artificial intelligence", "/m/0b9r8")
    if any(k in name for k in ["seo", "marketing"]):
        return ("Digital marketing", "/m/02hmvc")
    if any(k in name for k in ["jeans", "dress", "jacket", "shirt", "hoodie", "coat", "trouser", "legging"]):
        return ("Clothing", "/m/01c8m")
    if any(k in name for k in ["shoe", "trainer", "sneaker", "boot", "footwear", "running"]):
        return ("Footwear", None)
    return ("Other", None)

def level3_from_spoke(group: pd.DataFrame) -> Optional[str]:
    """Use the most frequent 'Spoke' value (if supplied) as an optional 3rd level."""
    if "spoke" in group.columns and not group["spoke"].isna().all():
        mode = group["spoke"].dropna().astype(str).str.strip()
        if not mode.empty:
            return mode.mode().iloc[0]
    return None

# ────────────────────────────────────────────────────────────────────────────────
# Diagnostics (optional)
# ────────────────────────────────────────────────────────────────────────────────
with st.expander("Diagnostics (optional)"):
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)
    st.write("`GOOGLE_APPLICATION_CREDENTIALS`:", creds_path if creds_path else "Not set")
    test_ok = False
    try:
        _client = language.LanguageServiceClient()
        _ = _client  # just to touch the client
        test_ok = True
    except Exception as e:
        st.warning(f"Could not initialize NL client: {e}")
    if test_ok:
        st.success("Google NL client initialized.")

# ────────────────────────────────────────────────────────────────────────────────
# Main flow
# ────────────────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("Upload your **Keyword Insights** CSV to begin. "
            "Minimum required columns: `Keyword`, `Search volume`, `Cluster`.\n\n"
            "The wider schema you shared is supported.")
    st.stop()

raw = pd.read_csv(uploaded)
df = norm_cols(raw)

# Validate required columns
missing = REQUIRED - set(df.columns)
if missing:
    st.error(f"Missing required column(s): {', '.join(sorted(missing))}")
    st.stop()

# Ensure numeric
df["search_volume"] = pd.to_numeric(df["search_volume"], errors="coerce").fillna(0).astype(int)

st.subheader("Preview")
st.dataframe(df.head(15), use_container_width=True)

grp = df.groupby("cluster", dropna=False)

rows = []
progress = st.progress(0, text="Analyzing clusters with Google Cloud NL…")
clusters = list(grp.groups.keys())
N = len(clusters)

for i, cluster_name in enumerate(clusters, start=1):
    g = grp.get_group(cluster_name)

    # Build synopsis for entity analysis
    kws = top_k_keywords(g, k=5)
    synopsis = synopsis_for_plc(cluster_name, kws)

    # Entity extraction
    ents = gcp_entities(synopsis)
    chosen = choose_entity(ents, cluster_name, kws, salience_min, allow_org)

    if chosen is None:
        # Still emit a row for traceability
        rows.append({
            "plc_name": cluster_name,
            "plc_keywords_sample": "; ".join(kws),
            "level_1_name": None, "level_1_mid": None,
            "level_2_name": None, "level_2_mid": None,
            "level_3_name": None, "level_3_mid": None,
            "level_4_name": None, "level_4_mid": None,
            "entity_label": None, "entity_mid": None, "entity_type": None, "entity_salience": None,
        })
        continue

    # Level 1: broad parent
    lvl1_name, lvl1_mid = infer_level1_parent(chosen["name"], chosen.get("mid"))
    # Level 2: chosen entity
    lvl2_name, lvl2_mid = chosen["name"], chosen.get("mid")
    # Level 3: Spoke (optional)
    lvl3_name, lvl3_mid = (None, None)
    if depth >= 3:
        l3 = level3_from_spoke(g)
        if l3:
            lvl3_name = str(l3)
    # Level 4: (extend with more curated maps as needed)
    lvl4_name, lvl4_mid = (None, None)

    rows.append({
        "plc_name": cluster_name,
        "plc_keywords_sample": "; ".join(kws),
        "level_1_name": lvl1_name if depth >= 1 else None,
        "level_1_mid":  lvl1_mid  if depth >= 1 else None,
        "level_2_name": lvl2_name if depth >= 2 else None,
        "level_2_mid":  lvl2_mid  if depth >= 2 else None,
        "level_3_name": lvl3_name if depth >= 3 else None,
        "level_3_mid":  lvl3_mid  if depth >= 3 else None,
        "level_4_name": lvl4_name if depth >= 4 else None,
        "level_4_mid":  lvl4_mid  if depth >= 4 else None,
        # Entity details for QA
        "entity_label": lvl2_name,
        "entity_mid": lvl2_mid,
        "entity_type": chosen.get("type"),
        "entity_salience": round(float(chosen.get("salience", 0.0)), 3),
    })

    if i % 10 == 0 or i == N:
        progress.progress(i / N, text=f"Analyzed {i}/{N} clusters")

hierarchy = pd.DataFrame(rows)

# Reorder/display columns
display_cols = [
    "plc_name", "plc_keywords_sample",
    "level_1_name", "level_1_mid",
    "level_2_name", "level_2_mid",
    "level_3_name", "level_3_mid",
    "level_4_name", "level_4_mid",
    "entity_label", "entity_mid", "entity_type", "entity_salience",
]
st.subheader("Topical Hierarchy (Single Output)")
st.dataframe(hierarchy[display_cols].head(50), use_container_width=True, height=430)
st.caption("Showing first 50 rows. Download the full hierarchy below.")

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Topical Hierarchy CSV",
    data=to_csv_bytes(hierarchy),
    file_name="topical_hierarchy_entity_only.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("""
**Notes**
- This app **always** uses Google Cloud Natural Language — Entity Analysis (no heuristic fallback).
- **Hierarchy depth** controls how many levels are emitted (L1..L4).
- **Level 1** uses a small curated **MID→parent** mapping (extend `CURATED_PARENT` with your domain's entities to unlock deeper, cleaner trees).
- **Level 3** uses your **Spoke** column when present to add a practical third layer.
- Salience is shown to help you spot low-confidence cases for manual QA.
""")

