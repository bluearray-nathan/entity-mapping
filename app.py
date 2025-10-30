# app.py
import os
import json
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import language_v1 as language

# ───────────────────────────────────────────────
# Page Setup
# ───────────────────────────────────────────────
st.set_page_config(page_title="Topical Hierarchy — Google Entity–Aligned", layout="wide")
st.title("Topical Hierarchy — Google Entity–Aligned")
st.caption("Upload a CSV with **Keyword, Search volume, Cluster** to generate a topical hierarchy using Google Cloud Natural Language.")

# ───────────────────────────────────────────────
# Credentials
# ───────────────────────────────────────────────
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

# ───────────────────────────────────────────────
# Sidebar Options
# ───────────────────────────────────────────────
with st.sidebar:
    st.header("Options")
    depth = st.selectbox("Hierarchy depth", [1, 2, 3], index=2)
    salience_min = st.slider("Minimum entity salience", 0.0, 1.0, 0.35, 0.05)
    allow_org = st.checkbox("Allow ORGANIZATION entities", value=False)
    st.caption("Requires Google Cloud Natural Language API access.")

# ───────────────────────────────────────────────
# Upload CSV
# ───────────────────────────────────────────────
uploaded = st.file_uploader("Upload your CSV", type=["csv"])
REQUIRED = {"keyword", "search_volume", "cluster"}

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    return df.rename(columns=mapping)

# ───────────────────────────────────────────────
# Google Cloud NL Entity Analysis
# ───────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def gcp_entities(text: str) -> List[Dict]:
    client = language.LanguageServiceClient()
    doc = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
    resp = client.analyze_entities(document=doc)
    out = []
    for e in resp.entities:
        out.append({
            "name": e.name,
            "type": language.Entity.Type(e.type_).name,
            "salience": float(e.salience),
            "mid": e.metadata.get("mid") if e.metadata else None,
        })
    return out

# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────
def top_k_keywords(group: pd.DataFrame, k: int = 5) -> List[str]:
    temp = group.copy()
    temp["search_volume"] = pd.to_numeric(temp["search_volume"], errors="coerce").fillna(0)
    temp = temp.sort_values("search_volume", ascending=False)
    return [str(x) for x in temp["keyword"].tolist()[:k]]

def synopsis_for_cluster(cluster_name: str, kws: List[str]) -> str:
    return f"{cluster_name}. Related searches: " + "; ".join(kws)

def choose_entity(ents: List[Dict], cluster_name: str, kws: List[str],
                  salience_threshold: float, allow_org: bool) -> Optional[Dict]:
    if not ents:
        return None
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
        return 0.7 * e.get("salience", 0) + 0.3 * coverage

    with_mid = [e for e in filtered if e.get("mid")]
    with_mid.sort(key=score, reverse=True)
    without_mid = [e for e in filtered if not e.get("mid")]
    without_mid.sort(key=score, reverse=True)

    if with_mid and score(with_mid[0]) >= salience_threshold:
        return with_mid[0]
    if with_mid:
        return with_mid[0]
    if without_mid:
        if without_mid[0]["name"].strip().lower() == cluster_name.strip().lower() and len(without_mid) > 1:
            return without_mid[1]
        return without_mid[0]
    return None

# Broad parent mapping
CURATED_PARENT = {
    "/m/0b9r8": ("Artificial intelligence", "/m/0b9r8"),
    "/m/02qkm5n": ("Data analytics", "/m/02qkm5n"),
    "/m/025rsfk": ("Business intelligence", "/m/025rsfk"),
    "/m/05wkwv": ("Business analytics", "/m/05wkwv"),
    "/m/01c8m": ("Clothing", "/m/01c8m"),
    "/m/027b9": ("Jeans", "/m/027b9"),
    "/m/027b7": ("Dress", "/m/027b7"),
    "/m/01b7x": ("Jacket", "/m/01b7x"),
}

def infer_parent(entity_name: str, entity_mid: Optional[str]) -> Tuple[str, Optional[str]]:
    if entity_mid in CURATED_PARENT:
        return CURATED_PARENT[entity_mid]
    name = (entity_name or "").lower()
    if any(k in name for k in ["analytics", "intelligence", "ai", "machine learning"]):
        return ("Artificial intelligence", "/m/0b9r8")
    if any(k in name for k in ["jeans", "dress", "shirt", "hoodie", "coat"]):
        return ("Clothing", "/m/01c8m")
    if any(k in name for k in ["shoe", "trainer", "sneaker", "boot"]):
        return ("Footwear", None)
    return ("Other", None)

def derive_facet_from_keywords(kws: list[str]) -> Optional[str]:
    text = " ".join(kws).lower()
    facets = []
    if re.search(r"\bmen|mens|men's\b", text): facets.append("Men")
    if re.search(r"\bwomen|womens|women's\b", text): facets.append("Women")
    if re.search(r"\b(kids|boys|girls|children)\b", text): facets.append("Kids")
    for m in ["leather", "denim", "cotton", "linen", "wool", "organic"]:
        if m in text: facets.append(m.title())
    label = " / ".join(dict.fromkeys(facets))
    return label or None

# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
if uploaded is None:
    st.info("Upload a CSV with **Keyword, Search volume, Cluster** to begin.")
    st.stop()

df = pd.read_csv(uploaded)
df = norm_cols(df)
missing = REQUIRED - set(df.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

df["search_volume"] = pd.to_numeric(df["search_volume"], errors="coerce").fillna(0).astype(int)
st.write("### Preview")
st.dataframe(df.head(10), use_container_width=True)

grp = df.groupby("cluster", dropna=False)
rows = []
progress = st.progress(0, text="Analyzing clusters…")
clusters = list(grp.groups.keys())

for i, cluster_name in enumerate(clusters, start=1):
    g = grp.get_group(cluster_name)
    kws = top_k_keywords(g)
    synopsis = synopsis_for_cluster(cluster_name, kws)

    ents = gcp_entities(synopsis)
    chosen = choose_entity(ents, cluster_name, kws, salience_min, allow_org)

    if not chosen:
        chosen = {"name": cluster_name, "mid": None, "salience": 0, "type": "OTHER"}

    lvl1_name, lvl1_mid = infer_parent(chosen["name"], chosen.get("mid"))
    lvl2_name, lvl2_mid = chosen["name"], chosen.get("mid")
    lvl3_name = derive_facet_from_keywords(kws) if depth >= 3 else None

    rows.append({
        "plc_name": cluster_name,
        "plc_keywords_sample": "; ".join(kws),
        "level_1_name": lvl1_name if depth >= 1 else None,
        "level_1_mid": lvl1_mid if depth >= 1 else None,
        "level_2_name": lvl2_name if depth >= 2 else None,
        "level_2_mid": lvl2_mid if depth >= 2 else None,
        "level_3_name": lvl3_name if depth >= 3 else None,
        "entity_salience": round(float(chosen.get("salience", 0)), 3),
        "entity_type": chosen.get("type"),
    })

    if i % 10 == 0 or i == len(clusters):
        progress.progress(i / len(clusters), text=f"Processed {i}/{len(clusters)} clusters")

hierarchy = pd.DataFrame(rows)

st.subheader("Topical Hierarchy")
st.dataframe(hierarchy.head(50), use_container_width=True, height=400)
st.download_button(
    "Download Topical Hierarchy CSV",
    data=hierarchy.to_csv(index=False).encode("utf-8"),
    file_name="topical_hierarchy_entity_only.csv",
    mime="text/csv"
)


