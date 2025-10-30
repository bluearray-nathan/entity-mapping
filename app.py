# app.py
import os, json, re
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import requests

# Optional (enabled by default if available)
USE_GCP = True
try:
    from google.cloud import language_v1 as language
except Exception:
    USE_GCP = False

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Topical Hierarchy — NL + KG", layout="wide")
st.title("Topical Hierarchy — Google NL + Knowledge Graph")
st.caption("Upload a CSV with **Keyword, Search volume, Cluster**. Choose a sector preset. The app maps each page-level cluster to Google entities (MIDs) and emits a single hierarchy table. Includes KG Search fallback for missing MIDs.")

# ─────────────────────────────────────────────────────────
# Credentials (Secrets → gcp / gcp_json for NL; kg_api_key for KG Search)
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

if USE_GCP and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    _maybe_write_sa_from_secrets()

# ─────────────────────────────────────────────────────────
# Sidebar options
# ─────────────────────────────────────────────────────────
SECTORS = ["Auto-detect (generic)", "Property", "Retail", "Finance", "B2B Analytics", "Travel"]

with st.sidebar:
    st.header("Options")
    selected_sector = st.selectbox("Sector preset", SECTORS, index=0)
    depth = st.selectbox("Hierarchy depth", [1, 2, 3], index=2)
    salience_min = st.slider("Minimum NL salience", 0.0, 1.0, 0.15, 0.05)
    allow_org = st.checkbox("Allow ORGANIZATION entities (brands/companies)", value=True)
    use_kg = st.checkbox("Enable Knowledge Graph fallback (resolve missing MIDs)", value=True)
    st.caption("KG fallback requires `kg_api_key` in Secrets.")
    st.markdown("---")
    st.caption("Input must be exactly: Keyword, Search volume, Cluster.")

uploaded = st.file_uploader("Upload CSV (columns: Keyword, Search volume, Cluster)", type=["csv"])
REQUIRED = {"keyword", "search_volume", "cluster"}

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})

# ─────────────────────────────────────────────────────────
# Sector hints & L3 facet rules
# ─────────────────────────────────────────────────────────
SECTOR_HINTS = {
    "Property": ["property", "real estate", "rentals", "for sale", "lettings", "housing"],
    "Retail": ["retail", "ecommerce", "apparel", "clothing", "footwear", "materials"],
    "Finance": ["finance", "fundraising", "platforms", "banking", "investing", "payments"],
    "B2B Analytics": ["analytics", "business intelligence", "dashboards", "data", "software"],
    "Travel": ["travel", "tourism", "destinations", "hotels", "airlines"],
}

def sector_context_tokens(sector: str) -> List[str]:
    return SECTOR_HINTS.get(sector, [])

# Property facets (bedrooms/tenure)
def facet_property(kws: List[str]) -> Optional[str]:
    text = " ".join(kws).lower()
    # bedrooms
    m = re.search(r"(\d+)\s*(bed|beds|bedroom|bedrooms)\b", text)
    if m:
        return f"{m.group(1)} Bedroom"
    if "studio" in text:
        return "0 Bedroom"
    # tenure
    if "to rent" in text or "for rent" in text or "rent " in text:
        return "To Rent"
    if "for sale" in text or "to buy" in text or "buy " in text:
        return "For Sale"
    return None

# Retail facets (gender/material/style)
def facet_retail(kws: List[str]) -> Optional[str]:
    t = " ".join(kws).lower()
    parts = []
    if re.search(r"\bmen|mens|men's\b", t): parts.append("Men")
    if re.search(r"\bwomen|womens|women's\b", t): parts.append("Women")
    if re.search(r"\b(kids|boys|girls|children)\b", t): parts.append("Kids")
    for m in ["leather","denim","cotton","linen","wool","synthetic","vegan","suede"]:
        if m in t: parts.append(m.title())
    for s in ["slim fit","regular fit","oversized","maxi","midi","high waisted","puffer","chelsea"]:
        if s in t: parts.append(s.title())
    return " / ".join(dict.fromkeys(parts)) or None

# Finance facets (crowdfunding subtypes, etc.)
def facet_finance(kws: List[str]) -> Optional[str]:
    t = " ".join(kws).lower()
    if "equity" in t: return "Equity crowdfunding"
    if "donation" in t or "donor" in t: return "Donation-based crowdfunding"
    if "reward" in t or "kickstarter" in t or "indiegogo" in t: return "Reward-based crowdfunding"
    return None

# B2B Analytics facets (light)
def facet_b2b_analytics(kws: List[str]) -> Optional[str]:
    t = " ".join(kws).lower()
    if "dashboard" in t or "dashboards" in t: return "Dashboards"
    if "report" in t or "reporting" in t: return "Reporting"
    if "predictive" in t or "forecast" in t: return "Predictive"
    return None

# Travel facets (location/intent-lite)
def facet_travel(kws: List[str]) -> Optional[str]:
    t = " ".join(kws).lower()
    for intent in ["flights", "hotels", "things to do", "itinerary", "resorts"]:
        if intent in t: return intent.title()
    return None

def facet_for_sector(sector: str, kws: List[str]) -> Optional[str]:
    if sector == "Property": return facet_property(kws)
    if sector == "Retail": return facet_retail(kws)
    if sector == "Finance": return facet_finance(kws)
    if sector == "B2B Analytics": return facet_b2b_analytics(kws)
    if sector == "Travel": return facet_travel(kws)
    return None

# ─────────────────────────────────────────────────────────
# NL & KG utilities
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def gcp_entities(text: str) -> List[Dict]:
    if not USE_GCP:
        return []
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

def resolve_mid_via_kg(name: str) -> Optional[str]:
    if not use_kg or "kg_api_key" not in st.secrets:
        return None
    try:
        r = requests.get(
            "https://kgsearch.googleapis.com/v1/entities:search",
            params={"query": name, "limit": 1, "key": st.secrets["kg_api_key"]},
            timeout=8,
        )
        if r.ok:
            items = r.json().get("itemListElement", [])
            if items:
                return items[0]["result"].get("@id")
    except Exception:
        return None
    return None

def choose_entity(ents: List[Dict], cluster_name: str, kws: List[str],
                  salience_threshold: float, allow_org: bool) -> Optional[Dict]:
    if not ents:
        return None
    # filter types
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

    if with_mid and score(with_mid[0]) >= salience_threshold:
        return with_mid[0]
    if with_mid:
        return with_mid[0]
    if without_mid:
        top = without_mid[0]
        if top["name"].strip().lower() == cluster_name.strip().lower() and len(without_mid) > 1:
            return without_mid[1]
        return top
    return None

# Curated MID → Level-1 parent
CURATED_PARENT = {
    # Finance
    "/m/07qwrb7": ("Finance", "/m/03fp41"),   # Crowdfunding -> Finance
    # Retail
    "/m/0h7f_": ("Retail", None),             # Running shoe
    "/m/027b9": ("Retail", None),             # Jeans
    "/m/027b7": ("Retail", None),             # Dress
    # B2B/Analytics
    "/m/02qkm5n": ("B2B Analytics", None),    # Data analytics
    "/m/025rsfk": ("B2B Analytics", None),    # Business intelligence
    # Travel
    # (Add destinations if you want them to map under Travel rather than Geography)
}

def infer_level1_parent(entity_name: str, entity_mid: Optional[str], sector: str) -> Tuple[str, Optional[str]]:
    # 1) curated map
    if entity_mid in CURATED_PARENT:
        return CURATED_PARENT[entity_mid]
    # 2) sector override
    if sector in SECTORS and sector != "Auto-detect (generic)":
        return (sector, None)
    # 3) heuristic fallback
    name = (entity_name or "").lower()
    if any(k in name for k in ["analytics", "intelligence", "dashboard", "data"]):
        return ("B2B Analytics", None)
    if any(k in name for k in ["shoe","jeans","dress","hoodie","jacket","apparel","clothing"]):
        return ("Retail", None)
    if any(k in name for k in ["flight","hotel","tourism","destination","airport"]):
        return ("Travel", None)
    if any(k in name for k in ["mortgage","loan","banking","investing","finance","crowdfunding"]):
        return ("Finance", None)
    if any(k in name for k in ["property","apartment","flat","house","estate"]):
        return ("Property", None)
    return ("Other", None)

# ─────────────────────────────────────────────────────────
# Synopsis & keyword utilities
# ─────────────────────────────────────────────────────────
def top_k_keywords(group: pd.DataFrame, k: int = 6) -> List[str]:
    temp = group.copy()
    temp["search_volume"] = pd.to_numeric(temp["search_volume"], errors="coerce").fillna(0)
    temp = temp.sort_values("search_volume", ascending=False)
    return [str(x) for x in temp["keyword"].tolist()[:k]]

def synopsis_for_cluster(cluster_name: str, kws: List[str], sector: str, max_k=6) -> str:
    cluster_l = cluster_name.strip().lower()
    uniq = [k for k in kws if k.strip().lower() != cluster_l][:max_k] or kws[:3]
    hints = sector_context_tokens(sector if sector != "Auto-detect (generic)" else "")
    context = f". Context: " + ", ".join(hints) if hints else ""
    return f"{cluster_name}. Related searches: " + "; ".join(uniq) + context

# ─────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────
with st.expander("Diagnostics (optional)"):
    st.write("NL client:", "available" if USE_GCP else "not available")
    st.write("KG fallback:", "enabled" if (use_kg and "kg_api_key" in st.secrets) else "disabled")

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
if uploaded is None:
    st.info("Upload a CSV with exactly **Keyword, Search volume, Cluster**.")
    st.stop()

df = pd.read_csv(uploaded)
df = norm_cols(df)

missing = REQUIRED - set(df.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

df["search_volume"] = pd.to_numeric(df["search_volume"], errors="coerce").fillna(0).astype(int)

st.subheader("Preview")
st.dataframe(df.head(12), use_container_width=True)

grouped = df.groupby("cluster", dropna=False)
clusters = list(grouped.groups.keys())

rows = []
progress = st.progress(0, text="Analyzing clusters…")

for i, cluster_name in enumerate(clusters, start=1):
    g = grouped.get_group(cluster_name)
    kws = top_k_keywords(g, k=6)

    # 1) Build synopsis (sector preset → adds context hints)
    synopsis = synopsis_for_cluster(cluster_name, kws, selected_sector)

    # 2) NL pass
    ents = gcp_entities(synopsis) if USE_GCP else []
    chosen = choose_entity(ents, cluster_name, kws, salience_min, allow_org)

    # 3) Retry with extra hints if no MID/name weak
    if (not chosen) or (not chosen.get("mid")):
        # add a second, gentle context tail for retry
        retry_hints = []
        t = " ".join(kws).lower()
        if selected_sector == "Retail" and any(x in t for x in ["shoe","trainer","sneaker","boot","jeans","dress","hoodie","jacket"]):
            retry_hints += ["apparel", "clothing", "footwear"]
        if selected_sector == "Finance" and "crowd" in t:
            retry_hints += ["finance", "fundraising", "platforms"]
        if selected_sector == "Property" and any(x in t for x in ["flat","house","studio","rent","sale","bedroom"]):
            retry_hints += ["property", "real estate", "rentals", "for sale"]
        enriched = synopsis + (". Context: " + "; ".join(dict.fromkeys(retry_hints)) if retry_hints else "")
        ents2 = gcp_entities(enriched) if USE_GCP else []
        chosen2 = choose_entity(ents2, cluster_name, kws, salience_min, allow_org)
        if chosen2:
            chosen = chosen2

    # 4) KG fallback if still missing MID
    if chosen and not chosen.get("mid"):
        mid = resolve_mid_via_kg(chosen["name"])
        if mid:
            chosen["mid"] = mid

    # 5) Build hierarchy
    if not chosen:
        chosen = {"name": cluster_name, "mid": None, "salience": 0.0, "type": "OTHER"}

    lvl1_name, lvl1_mid = infer_level1_parent(chosen["name"], chosen.get("mid"), selected_sector)
    lvl2_name, lvl2_mid = chosen["name"], chosen.get("mid")

    lvl3_name, lvl3_mid = (None, None)
    if depth >= 3:
        # sector facet (no MID)
        lvl3_name = facet_for_sector(selected_sector, kws)

    rows.append({
        "plc_name": cluster_name,
        "plc_keywords_sample": "; ".join(kws),
        "level_1_name": lvl1_name if depth >= 1 else None,
        "level_1_mid": lvl1_mid if depth >= 1 else None,
        "level_2_name": lvl2_name if depth >= 2 else None,
        "level_2_mid": lvl2_mid if depth >= 2 else None,
        "level_3_name": lvl3_name if depth >= 3 else None,
        "level_3_mid": lvl3_mid if depth >= 3 else None,
        # QA fields
        "entity_type": chosen.get("type"),
        "entity_salience": round(float(chosen.get("salience", 0.0)), 3),
    })

    if i % 10 == 0 or i == len(clusters):
        progress.progress(i / len(clusters), text=f"Processed {i}/{len(clusters)} clusters")

hier = pd.DataFrame(rows)

st.subheader("Topical Hierarchy (single output)")
show_cols = [
    "plc_name", "plc_keywords_sample",
    "level_1_name", "level_1_mid",
    "level_2_name", "level_2_mid",
    "level_3_name", "level_3_mid",
    "entity_type", "entity_salience"
]
st.dataframe(hier[show_cols].head(50), use_container_width=True, height=420)

st.download_button(
    "Download Topical Hierarchy CSV",
    data=hier.to_csv(index=False).encode("utf-8"),
    file_name="topical_hierarchy_nl_kg.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("""
**What changed**
- **Sector presets** inject context for better NL linking and sector-specific **Level-3 facets**.
- **Knowledge Graph fallback** fills **MIDs** when NL returns only a name.
- **Lower salience default (0.15)** to capture legitimate MIDs in short texts.
- Still one clean output table for client-friendly use.
""")





