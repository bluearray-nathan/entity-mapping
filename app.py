# app.py
import os
import json
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import language_v1 as language

# ─────────────────────────────────────────────────────────
# Streamlit page config
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Topical Hierarchy — Google Entity–Aligned", layout="wide")
st.title("Topical Hierarchy — Google Entity–Aligned")
st.caption("Upload a CSV with **Keyword, Search volume, Cluster**. The app maps each page-level Cluster to Google Knowledge Graph entities (MIDs) and outputs a single topical hierarchy table.")

# ─────────────────────────────────────────────────────────
# Credentials via Streamlit Secrets (or pre-set env var)
# ─────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Options")
    depth = st.selectbox("Hierarchy depth", [1, 2, 3], index=2)
    # Slightly lower default threshold to help entity linking in niche B2B terms
    salience_min = st.slider("Minimum entity salience", 0.0, 1.0, 0.25, 0.05)
    allow_org = st.checkbox("Allow ORGANIZATION entities (brands, companies)", value=False,
                            help="Turn on if many clusters are brand-led; improves MID hit rate for brandy phrases.")
    st.markdown("---")
    st.caption("Requires Google Cloud Natural Language API access.")

# ─────────────────────────────────────────────────────────
# Upload CSV (3 columns only)
# ─────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV (columns: Keyword, Search volume, Cluster)", type=["csv"])
REQUIRED = {"keyword", "search_volume", "cluster"}

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    return df.rename(columns=mapping)

# ─────────────────────────────────────────────────────────
# Google Cloud NL — Entity Analysis
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def gcp_entities(text: str) -> List[Dict]:
    client = language.LanguageServiceClient()
    doc = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
    resp = client.analyze_entities(document=doc)
    out = []
    for e in resp.entities:
        out.append({
            "name": e.name,
            "type": language.Entity.Type(e.type_).name,  # PERSON/ORGANIZATION/LOCATION/EVENT/WORK_OF_ART/CONSUMER_GOOD/OTHER
            "salience": float(e.salience),
            "mid": e.metadata.get("mid") if e.metadata else None,
            "wikipedia_url": e.metadata.get("wikipedia_url") if e.metadata else None,
        })
    return out

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def top_k_keywords(group: pd.DataFrame, k: int = 6) -> List[str]:
    temp = group.copy()
    temp["search_volume"] = pd.to_numeric(temp["search_volume"], errors="coerce").fillna(0)
    temp = temp.sort_values("search_volume", ascending=False)
    return [str(x) for x in temp["keyword"].tolist()[:k]]

# === Option A: Richer synopsis (avoid echoing cluster; add variety) ===
def synopsis_for_cluster(cluster_name: str, kws: list[str], max_k=6) -> str:
    # Deduplicate keywords that are identical to the cluster text
    cluster_l = cluster_name.strip().lower()
    uniq = [k for k in kws if k.strip().lower() != cluster_l]
    # If all keywords mirror the cluster, still keep a few to give NL some text
    uniq = uniq[:max_k] if uniq else kws[:3]
    return f"{cluster_name}. Related searches: " + "; ".join(uniq)

# === Option B: Chooser that prefers KG-linked entities (with MID) ===
def choose_entity(ents, cluster_name, kws, salience_threshold=0.25, allow_org=False):
    if not ents:
        return None

    # Filter undesirable types
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

    text = (cluster_name + " " + " ".join(kws)).lower()

    def score(e):
        name = e["name"].lower()
        coverage = 1.0 if name in text else 0.0
        return 0.7 * float(e.get("salience", 0.0)) + 0.3 * coverage

    with_mid = sorted([e for e in filtered if e.get("mid")], key=score, reverse=True)
    without_mid = sorted([e for e in filtered if not e.get("mid")], key=score, reverse=True)

    # Prefer with MID (even slightly below threshold), then fall back
    if with_mid and score(with_mid[0]) >= salience_threshold:
        return with_mid[0]
    if with_mid:
        return with_mid[0]

    # Last resort: avoid returning the exact cluster text if possible
    if without_mid:
        top = without_mid[0]
        if top["name"].strip().lower() == cluster_name.strip().lower() and len(without_mid) > 1:
            return without_mid[1]
        return top
    return None

# Broad parent mapping (extend with your domain’s entities to improve Level-1)
CURATED_PARENT = {
    # B2B / Analytics examples
    "/m/0b9r8": ("Artificial intelligence", "/m/0b9r8"),
    "/m/02qkm5n": ("Data analytics", "/m/02qkm5n"),
    "/m/025rsfk": ("Business intelligence", "/m/025rsfk"),
    "/m/05wkwv": ("Business analytics", "/m/05wkwv"),
    # Retail examples
    "/m/01c8m": ("Clothing", "/m/01c8m"),
    "/m/027b9": ("Jeans", "/m/027b9"),
    "/m/027b7": ("Dress", "/m/027b7"),
    "/m/01b7x": ("Jacket", "/m/01b7x"),
    "/m/02fg3h": ("Hoodie", "/m/02fg3h"),
    "/m/0h7f_": ("Running shoe", "/m/0h7f_"),
}

def infer_level1_parent(entity_name: str, entity_mid: Optional[str]) -> Tuple[str, Optional[str]]:
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

# Optional: keyword-derived facet for Level-3 (no MID)
def derive_facet_from_keywords(kws: list[str]) -> Optional[str]:
    text = " ".join(kws).lower()
    facets = []
    if re.search(r"\bmen|mens|men's\b", text): facets.append("Men")
    if re.search(r"\bwomen|womens|women's\b", text): facets.append("Women")
    if re.search(r"\b(kids|boys|girls|children)\b", text): facets.append("Kids")
    for m in ["leather", "denim", "cotton", "linen", "wool", "organic", "vegan"]:
        if m in text: facets.append(m.title())
    label = " / ".join(dict.fromkeys(facets))
    return label or None

# ─────────────────────────────────────────────────────────
# Diagnostics (optional)
# ─────────────────────────────────────────────────────────
with st.expander("Diagnostics (optional)"):
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)
    st.write("`GOOGLE_APPLICATION_CREDENTIALS`:", creds_path if creds_path else "Not set")
    try:
        _ = language.LanguageServiceClient()
        st.success("Google NL client initialized.")
    except Exception as e:
        st.warning(f"Could not initialize NL client: {e}")

# ─────────────────────────────────────────────────────────
# Main flow
# ─────────────────────────────────────────────────────────
if uploaded is None:
    st.info("Upload a CSV with exactly these columns: **Keyword, Search volume, Cluster**.")
    st.stop()

df = pd.read_csv(uploaded)
df = norm_cols(df)

missing = REQUIRED - set(df.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

df["search_volume"] = pd.to_numeric(df["search_volume"], errors="coerce").fillna(0).astype(int)

st.subheader("Preview")
st.dataframe(df.head(15), use_container_width=True)

grp = df.groupby("cluster", dropna=False)
rows = []
clusters = list(grp.groups.keys())
progress = st.progress(0, text="Analyzing clusters…")

for i, cluster_name in enumerate(clusters, start=1):
    g = grp.get_group(cluster_name)
    kws = top_k_keywords(g, k=6)

    # Build synopsis (Option A)
    synopsis = synopsis_for_cluster(cluster_name, kws)

    # First pass
    ents = gcp_entities(synopsis)
    chosen = choose_entity(ents, cluster_name, kws, salience_min, allow_org)

    # Retry once with enriched context if no MID (Option B — retry)
    if not chosen or not chosen.get("mid"):
        hint_tokens = []
        text_all = " ".join(kws).lower()
        if any(t in text_all for t in ["jeans","dress","jacket","shirt","hoodie","coat","trousers","leggings","blazer"]):
            hint_tokens += ["clothing", "apparel"]
        if any(t in text_all for t in ["shoe","trainer","sneaker","boot","running"]):
            hint_tokens += ["footwear"]
        if any(t in text_all for t in ["analytics","intelligence","bi","dashboard","predictive","etl","warehouse"]):
            hint_tokens += ["data analytics"]
        if hint_tokens:
            enriched = synopsis + ". Context: " + "; ".join(dict.fromkeys(hint_tokens))
            ents2 = gcp_entities(enriched)
            chosen2 = choose_entity(ents2, cluster_name, kws, salience_min, allow_org)
            chosen = chosen2 or chosen

    # Populate hierarchy rows
    if not chosen:
        chosen = {"name": cluster_name, "mid": None, "salience": 0.0, "type": "OTHER"}

    lvl1_name, lvl1_mid = infer_level1_parent(chosen["name"], chosen.get("mid"))
    lvl2_name, lvl2_mid = chosen["name"], chosen.get("mid")

    lvl3_name, lvl3_mid = (None, None)
    if depth >= 3:
        # Keyword facet (no MID) — keeps Level-3 informative without extra API calls
        lvl3_name = derive_facet_from_keywords(kws)

    rows.append({
        "plc_name": cluster_name,
        "plc_keywords_sample": "; ".join(kws),
        "level_1_name": lvl1_name if depth >= 1 else None,
        "level_1_mid": lvl1_mid if depth >= 1 else None,
        "level_2_name": lvl2_name if depth >= 2 else None,
        "level_2_mid": lvl2_mid if depth >= 2 else None,
        "level_3_name": lvl3_name if depth >= 3 else None,
        "level_3_mid": lvl3_mid if depth >= 3 else None,
        "entity_label": lvl2_name,
        "entity_mid": lvl2_mid,
        "entity_type": chosen.get("type"),
        "entity_salience": round(float(chosen.get("salience", 0.0)), 3),
    })

    if i % 10 == 0 or i == len(clusters):
        progress.progress(i / len(clusters), text=f"Processed {i}/{len(clusters)} clusters")

hierarchy = pd.DataFrame(rows)

st.subheader("Topical Hierarchy (single output)")
show_cols = [
    "plc_name", "plc_keywords_sample",
    "level_1_name", "level_1_mid",
    "level_2_name", "level_2_mid",
    "level_3_name", "level_3_mid",
    "entity_type", "entity_salience",
]
st.dataframe(hierarchy[show_cols].head(50), use_container_width=True, height=420)
st.download_button(
    "Download Topical Hierarchy CSV",
    data=hierarchy.to_csv(index=False).encode("utf-8"),
    file_name="topical_hierarchy_entity_only.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("""
**Notes**
- **Option A** applied: enriched synopses reduce duplicate/echo text and improve entity linking.
- **Option B** applied: chooser prefers entries with **MIDs**, and we **retry once** with extra domain hints if the first pass lacks a MID.
- Level-3 here is a **facet** derived from keywords (no MID). If you want **MIDs at Level-3**, run a second NL pass on a sub-synopsis or add a KG Search backfill.
""")



