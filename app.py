# app.py
import os, json, re
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import requests

# Try to import GCP NL; if not present we still run (KG-only mode)
USE_GCP = True
try:
    from google.cloud import language_v1 as language
except Exception:
    USE_GCP = False

# ─────────────────────────────────────────────────────────
# Streamlit page config
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Topical Hierarchy — NL + KG", layout="wide")
st.title("Topical Hierarchy — Google NL + Knowledge Graph")
st.caption("Upload a CSV with **Keyword, Search volume, Cluster**. Sector presets add context; KG fallback resolves MIDs and canonical names.")

# ─────────────────────────────────────────────────────────
# Credentials: accept gcp_json OR [gcp] table in secrets
# ─────────────────────────────────────────────────────────
def _ensure_gcp_credentials():
    """Write a service account JSON to a temp file from secrets, supporting both gcp_json and [gcp]."""
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        return
    sa_info = None
    if "gcp_json" in st.secrets:
        try:
            sa_info = json.loads(st.secrets["gcp_json"])
        except Exception:
            # If gcp_json is already the raw object (Streamlit Cloud often parses), use it directly
            sa_info = dict(st.secrets["gcp_json"])
    elif "gcp" in st.secrets:
        sa_info = dict(st.secrets["gcp"])  # table form (your current setup)
    if not sa_info:
        return
    key_path = "/tmp/gcp-sa.json"
    with open(key_path, "w") as f:
        json.dump(sa_info, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

if USE_GCP:
    _ensure_gcp_credentials()

# ─────────────────────────────────────────────────────────
# Sidebar options
# ─────────────────────────────────────────────────────────
SECTORS = ["Auto-detect (generic)", "Property", "Retail", "Finance", "B2B Analytics", "Travel"]
with st.sidebar:
    st.header("Options")
    selected_sector = st.selectbox("Sector preset", SECTORS, index=0)
    depth = st.selectbox("Hierarchy depth", [1, 2, 3], index=2)
    salience_min = st.slider("Minimum NL salience", 0.0, 1.0, 0.15, 0.05)
    allow_org = st.checkbox("Allow ORGANIZATION (brands/companies)", value=True)
    use_kg = st.checkbox("Enable Knowledge Graph fallback (resolve missing MIDs)", value=True,
                         help="Requires `kg_api_key` in secrets.toml")
    st.caption("Input must be exactly: Keyword, Search volume, Cluster.")

# ─────────────────────────────────────────────────────────
# Sector hints & facets
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

def facet_property(kws: List[str]) -> Optional[str]:
    t = " ".join(kws).lower()
    m = re.search(r"(\d+)\s*(bed|beds|bedroom|bedrooms)\b", t)
    if m: return f"{m.group(1)} Bedroom"
    if "studio" in t: return "0 Bedroom"
    if any(x in t for x in ["to rent", "for rent", " rent "]): return "To Rent"
    if any(x in t for x in ["for sale", "to buy", " buy "]): return "For Sale"
    return None

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

def facet_finance(kws: List[str]) -> Optional[str]:
    t = " ".join(kws).lower()
    if "equity" in t: return "Equity crowdfunding"
    if "donation" in t or "donor" in t: return "Donation-based crowdfunding"
    if "reward" in t or "kickstarter" in t or "indiegogo" in t: return "Reward-based crowdfunding"
    return None

def facet_b2b_analytics(kws: List[str]) -> Optional[str]:
    t = " ".join(kws).lower()
    if "dashboard" in t: return "Dashboards"
    if "report" in t or "reporting" in t: return "Reporting"
    if "predictive" in t or "forecast" in t: return "Predictive"
    return None

def facet_travel(kws: List[str]) -> Optional[str]:
    t = " ".join(kws).lower()
    for intent in ["flights","hotels","things to do","itinerary","resorts"]:
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
# NL + KG utilities
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

def resolve_mid_and_name_via_kg(query_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (mid, canonical_name) using KG Search; None/None if disabled or no hit."""
    key = st.secrets.get("kg_api_key")
    if not (use_kg and key):
        return None, None
    try:
        r = requests.get(
            "https://kgsearch.googleapis.com/v1/entities:search",
            params={"query": query_name, "limit": 1, "key": key},
            timeout=8,
        )
        if r.ok:
            items = r.json().get("itemListElement", [])
            if items:
                result = items[0]["result"]
                return result.get("@id"), result.get("name")
    except Exception:
        pass
    return None, None

def choose_entity(ents: List[Dict], cluster_name: str, kws: List[str],
                  salience_threshold: float, allow_org: bool) -> Optional[Dict]:
    if not ents:
        return None
    filtered = []
    for e in ents:
        et = e.get("type", "OTHER")
        if et in {"PERSON","NUMBER","DATE","ADDRESS"}:
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
        return 0.7*float(e.get("salience",0.0)) + 0.3*coverage

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

# Curated MID → Level-1 parent (extend as needed)
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
}

def infer_level1_parent(entity_name: str, entity_mid: Optional[str], sector: str) -> Tuple[str, Optional[str]]:
    if entity_mid in CURATED_PARENT:
        return CURATED_PARENT[entity_mid]
    if sector in SECTORS and sector != "Auto-detect (generic)":
        return (sector, None)
    name = (entity_name or "").lower()
    if any(k in name for k in ["analytics","intelligence","dashboard","data"]): return ("B2B Analytics", None)
    if any(k in name for k in ["shoe","jeans","dress","hoodie","jacket","apparel","clothing"]): return ("Retail", None)
    if any(k in name for k in ["flight","hotel","tourism","destination","airport"]): return ("Travel", None)
    if any(k in name for k in ["mortgage","loan","banking","investing","finance","crowdfunding"]): return ("Finance", None)
    if any(k in name for k in ["property","apartment","flat","house","estate"]): return ("Property", None)
    return ("Other", None)

# ─────────────────────────────────────────────────────────
# Synopsis & input helpers
# ─────────────────────────────────────────────────────────
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})

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
# Upload & validate
# ─────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV (columns: Keyword, Search volume, Cluster)", type=["csv"])
REQUIRED = {"keyword","search_volume","cluster"}

with st.expander("Diagnostics (optional)"):
    st.write("NL client:", "available" if USE_GCP else "not available")
    st.write("KG fallback:", "enabled" if (use_kg and "kg_api_key" in st.secrets) else "disabled")
    with st.expander("KG Debug"):
        st.write("Sidebar use_kg toggle:", use_kg)
        st.write("Has 'kg_api_key' in secrets:", "kg_api_key" in st.secrets)
        if "kg_api_key" in st.secrets:
            st.write("kg_api_key length:", len(st.secrets["kg_api_key"]))

if uploaded is None:
    st.info("Upload a CSV with exactly **Keyword, Search volume, Cluster**.")
    st.stop()

df = pd.read_csv(uploaded)
df = normalize_cols(df)
missing = REQUIRED - set(df.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

df["search_volume"] = pd.to_numeric(df["search_volume"], errors="coerce").fillna(0).astype(int)

st.subheader("Preview")
st.dataframe(df.head(12), use_container_width=True)

# ─────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────
grouped = df.groupby("cluster", dropna=False)
clusters = list(grouped.groups.keys())
rows = []
progress = st.progress(0, text="Analyzing clusters…")

for i, cluster_name in enumerate(clusters, start=1):
    g = grouped.get_group(cluster_name)
    kws = top_k_keywords(g, k=6)

    # Build synopsis with sector context
    synopsis = synopsis_for_cluster(cluster_name, kws, selected_sector)

    # 1) NL pass (if available)
    chosen = None
    ents = gcp_entities(synopsis) if USE_GCP else []
    if ents:
        chosen = choose_entity(ents, cluster_name, kws, salience_min, allow_org)

    # 2) Retry with extra hints if no MID
    if (not chosen) or (not chosen.get("mid")):
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
        chosen2 = choose_entity(ents2, cluster_name, kws, salience_min, allow_org) if ents2 else None
        if chosen2:
            chosen = chosen2

    resolution_source = "NL"
    # 3) KG fallback (now returns both MID and canonical name)
    if chosen and not chosen.get("mid"):
        mid, canonical = resolve_mid_and_name_via_kg(chosen["name"])
        if mid:
            chosen["mid"] = mid
            if canonical and canonical.strip() and canonical.lower() != chosen["name"].lower():
                chosen["name"] = canonical
            resolution_source = "KG_fallback"

    # If still nothing, try mapping the cluster text itself via KG (some clusters never link via NL)
    if (not chosen) and use_kg:
        mid, canonical = resolve_mid_and_name_via_kg(cluster_name)
        if mid:
            chosen = {"name": canonical or cluster_name, "mid": mid, "salience": 0.0, "type": "OTHER"}
            resolution_source = "KG_direct"

    if not chosen:
        chosen = {"name": cluster_name, "mid": None, "salience": 0.0, "type": "OTHER"}
        resolution_source = "None"

    # Build hierarchy levels
    lvl1_name, lvl1_mid = infer_level1_parent(chosen["name"], chosen.get("mid"), selected_sector)
    lvl2_name, lvl2_mid = chosen["name"], chosen.get("mid")

    lvl3_name, lvl3_mid = (None, None)
    if depth >= 3:
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
        "entity_type": chosen.get("type"),
        "entity_salience": round(float(chosen.get("salience", 0.0)), 3),
        "resolution_source": resolution_source,
    })

    if i % 10 == 0 or i == len(clusters):
        progress.progress(i / len(clusters), text=f"Processed {i}/{len(clusters)} clusters")

hier = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────
st.subheader("Topical Hierarchy (single output)")
show_cols = [
    "plc_name", "plc_keywords_sample",
    "level_1_name", "level_1_mid",
    "level_2_name", "level_2_mid",
    "level_3_name", "level_3_mid",
    "entity_type", "entity_salience", "resolution_source"
]
st.dataframe(hier[show_cols].head(50), use_container_width=True, height=440)

st.download_button(
    "Download Topical Hierarchy CSV",
    data=hier.to_csv(index=False).encode("utf-8"),
    file_name="topical_hierarchy_nl_kg.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("""
**What’s improved**
- **KG fallback** now fills **MID + canonical name**, so “crowd fund” becomes **Crowdfunding** with the correct MID.
- **Sector presets** inject domain context (better NL linking) and sector-specific Level-3 facets.
- Works with **`gcp_json` or `[gcp]`** secrets formats. See the Diagnostics expander to confirm keys are loaded.
""")






