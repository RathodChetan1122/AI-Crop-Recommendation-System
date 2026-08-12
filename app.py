# 🌱 AI-BASED CROP RECOMMENDATION SYSTEM
# Modern AgriTech Dashboard — Streamlit Application


import traceback
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# CONFIGURATION / CONSTANTS

MODEL_PATH = "crop_recommendation_model.pkl"
FEATURE_NAMES_PATH = "crop_feature_names.pkl"

# Metrics are fixed facts about the trained model — do not
# recompute or invent these here; they simply describe the
# model that was trained and saved to MODEL_PATH.
MODEL_INFO = {
    "algorithm": "Tuned Random Forest",
    "n_classes": 22,
    "n_features": 7,
    "test_samples": 440,
    "correct_predictions": 438,
    "incorrect_predictions": 2,
    "test_accuracy": 99.55,
    "cv_accuracy": 99.60,
    "params": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 1,
        "min_samples_split": 5,
    },
}

# Feature ranges observed in the training dataset (2200 rows).
DATASET_RANGES = {
    "N": (0.0, 140.0),
    "P": (5.0, 145.0),
    "K": (5.0, 205.0),
    "temperature": (8.825675, 43.675493),
    "humidity": (14.258040, 99.981876),
    "ph": (3.504752, 9.935091),
    "rainfall": (20.211267, 298.560117),
}

# Human-friendly labels / units for each feature.
FEATURE_META = {
    "N": {"label": "Nitrogen (N)", "unit": "", "group": "nutrients"},
    "P": {"label": "Phosphorus (P)", "unit": "", "group": "nutrients"},
    "K": {"label": "Potassium (K)", "unit": "", "group": "nutrients"},
    "ph": {"label": "Soil pH", "unit": "", "group": "chemistry"},
    "temperature": {"label": "Temperature", "unit": "°C", "group": "environment"},
    "humidity": {"label": "Humidity", "unit": "%", "group": "environment"},
    "rainfall": {"label": "Rainfall", "unit": "mm", "group": "environment"},
}

# Default / sample values shown when the app first loads and
# restored whenever the user presses "Reset".
DEFAULT_VALUES = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 20.88,
    "humidity": 82.0,
    "ph": 6.50,
    "rainfall": 202.94,
}

INPUT_KEY_PREFIX = "input_"

CROP_INFO = {
    "rice": {
        "emoji": "🌾",
        "description": (
            "A water-demanding cereal crop commonly associated "
            "with relatively humid and high-rainfall conditions."
        ),
        "advice": "Consider local water availability and seasonal rainfall before cultivation.",
    },
    "maize": {
        "emoji": "🌽",
        "description": (
            "A major cereal crop whose suitability depends on the "
            "combined soil and environmental conditions."
        ),
        "advice": "Consider local climate, soil fertility and water availability.",
    },
    "chickpea": {
        "emoji": "🫘",
        "description": (
            "A pulse crop whose recommendation depends on the "
            "combination of nutrient and environmental conditions."
        ),
        "advice": "Check local season and soil conditions before cultivation.",
    },
    "kidneybeans": {
        "emoji": "🫘",
        "description": (
            "A pulse crop selected when the measured conditions "
            "match patterns learned from the training data."
        ),
        "advice": "Verify local growing-season suitability.",
    },
    "pigeonpeas": {
        "emoji": "🫘",
        "description": (
            "A pulse crop whose recommendation is based on the "
            "combined soil and environmental measurements."
        ),
        "advice": "Consider local rainfall and seasonal conditions.",
    },
    "mothbeans": {
        "emoji": "🌱",
        "description": (
            "A pulse crop selected based on patterns learned "
            "from soil and environmental measurements."
        ),
        "advice": "Check local water availability before planting.",
    },
    "mungbean": {
        "emoji": "🌱",
        "description": (
            "A pulse crop recommended when the input conditions "
            "match learned patterns for mungbean."
        ),
        "advice": "Consider local temperature and rainfall conditions.",
    },
    "blackgram": {
        "emoji": "🌱",
        "description": (
            "A pulse crop whose prediction depends on all seven "
            "soil and environmental features."
        ),
        "advice": "Use the recommendation together with local agricultural advice.",
    },
    "lentil": {
        "emoji": "🌱",
        "description": (
            "A pulse crop recommended according to patterns "
            "learned from the training dataset."
        ),
        "advice": "Consider the local season before cultivation.",
    },
    "pomegranate": {
        "emoji": "🍎",
        "description": (
            "A fruit crop selected according to the combined "
            "soil and environmental conditions."
        ),
        "advice": "Consider long-term climate and irrigation availability.",
    },
    "banana": {
        "emoji": "🍌",
        "description": (
            "A fruit crop whose recommendation depends on "
            "the combination of the seven input features."
        ),
        "advice": "Consider water availability and local climate.",
    },
    "mango": {
        "emoji": "🥭",
        "description": (
            "A fruit crop selected when the measured conditions "
            "match patterns learned for mango."
        ),
        "advice": "Consider long-term climate suitability.",
    },
    "grapes": {
        "emoji": "🍇",
        "description": (
            "A fruit crop recommended according to patterns "
            "learned from the soil and environmental data."
        ),
        "advice": "Consider local climate and irrigation conditions.",
    },
    "watermelon": {
        "emoji": "🍉",
        "description": (
            "A crop whose prediction depends on the combined "
            "temperature, humidity, soil and rainfall conditions."
        ),
        "advice": "Consider water availability during the growing period.",
    },
    "muskmelon": {
        "emoji": "🍈",
        "description": (
            "A crop selected according to patterns learned "
            "from the seven model inputs."
        ),
        "advice": "Consider temperature and water availability.",
    },
    "apple": {
        "emoji": "🍎",
        "description": (
            "A fruit crop selected when the environmental and "
            "soil measurements resemble learned patterns."
        ),
        "advice": "Consider local climate suitability.",
    },
    "orange": {
        "emoji": "🍊",
        "description": (
            "A fruit crop whose prediction uses the combined "
            "soil and environmental measurements."
        ),
        "advice": "Consider local climate and irrigation conditions.",
    },
    "papaya": {
        "emoji": "🥭",
        "description": (
            "A fruit crop that can be recommended under suitable "
            "temperature, humidity, soil and rainfall conditions."
        ),
        "advice": "Consider local water availability and climate.",
    },
    "coconut": {
        "emoji": "🥥",
        "description": (
            "A crop whose recommendation can be associated with "
            "humid and rainfall-related environmental conditions."
        ),
        "advice": "Consider long-term rainfall and water availability.",
    },
    "cotton": {
        "emoji": "🌿",
        "description": (
            "A commercial crop whose prediction depends on the "
            "combined soil and environmental measurements."
        ),
        "advice": "Consider local climate and market conditions.",
    },
    "jute": {
        "emoji": "🌿",
        "description": (
            "A crop whose recommendation depends on moisture, "
            "rainfall, temperature and soil characteristics."
        ),
        "advice": "Consider local rainfall and soil conditions.",
    },
    "coffee": {
        "emoji": "☕",
        "description": (
            "A crop selected when the measured conditions resemble "
            "patterns learned for coffee."
        ),
        "advice": "Consider local climate, shade and water availability.",
    },
}

DEFAULT_CROP_INFO = {
    "emoji": "🌱",
    "description": "A crop selected according to patterns learned from the training data.",
    "advice": "Use this recommendation together with local agricultural advice.",
}



# PAGE CONFIGURATION

st.set_page_config(
    page_title="AI Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# CUSTOM CSS — AgriTech theme

st.markdown(
    """
    <style>
    :root, .stApp {
        color-scheme: light;
        --forest: #14532d;
        --forest-2: #166534;
        --green: #16a34a;
        --green-light: #22c55e;
        --cream: #faf8f3;
        --gray-light: #f1f5f9;
        --border: #e2e8f0;
        --teal: #0d9488;
        /* Required text colors */
        --text-dark: #17201A;
        --text-muted: #64748B;
    }

    /* ============================================================
       BASE — force a light surface + dark text everywhere.
       Widgets that must show white text (hero/result cards) set
       their own color explicitly below with higher specificity.
       ============================================================ */
    html, body, .stApp {
        background: var(--cream) !important;
        color: var(--text-dark);
    }

    .stApp, .stApp p, .stApp span, .stApp li, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stMarkdown, .stMarkdown p {
        color: var(--text-dark);
    }

    .block-container {
        max-width: 1300px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ---------------- Hero (white text on green — allowed) ---------------- */
    .hero {
        background: linear-gradient(120deg, var(--forest) 0%, var(--green) 100%);
        border-radius: 22px;
        padding: 2.2rem 2.6rem;
        color: #ffffff !important;
        box-shadow: 0 16px 40px rgba(20, 83, 45, 0.25);
        margin-bottom: 1.4rem;
        overflow: hidden;
    }
    .hero * { color: #ffffff !important; }
    .hero-eyebrow {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        font-size: 0.76rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        font-weight: 700;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.96;
        max-width: 680px;
        line-height: 1.55;
    }

    /* ---------------- Section headers ---------------- */
    .section-title {
        font-size: 1.3rem;
        font-weight: 750;
        color: var(--forest) !important;
        margin: 1.4rem 0 0.7rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-title * { color: var(--forest) !important; }
    .section-sub {
        color: var(--text-muted) !important;
        font-size: 0.92rem;
        margin-bottom: 0.9rem;
        margin-top: -0.4rem;
    }
    .group-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--teal) !important;
        margin-bottom: 0.5rem;
    }

    /* ---------------- Panels ---------------- */
    .panel {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.2rem 0.4rem 1.2rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        margin-bottom: 0.8rem;
    }

    /* ---------------- Quick stats ---------------- */
    .stat-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }
    .stat-value { font-size: 1.5rem; font-weight: 800; color: var(--forest-2) !important; }
    .stat-label { font-size: 0.82rem; color: var(--text-muted) !important; margin-top: 0.15rem; }

    /* ---------------- Result card (white text on green — allowed) ---------------- */
    .result-card {
        background: linear-gradient(135deg, var(--forest-2) 0%, var(--green) 100%);
        padding: 2rem;
        border-radius: 20px;
        color: #ffffff !important;
        text-align: center;
        box-shadow: 0 12px 30px rgba(22, 101, 52, 0.22);
        margin-top: 0.4rem;
        margin-bottom: 1.3rem;
    }
    .result-card * { color: #ffffff !important; }
    .result-label { font-size: 1rem; opacity: 0.92; letter-spacing: 0.03em; }
    .result-crop { font-size: 2.9rem; font-weight: 800; margin: 0.35rem 0; }
    .result-confidence { font-size: 1.05rem; opacity: 0.96; }

    /* ---------------- Crop cards ---------------- */
    .crop-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.2rem 1.2rem 1.4rem 1.2rem;
        text-align: center;
        border: 1px solid #dcfce7;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        min-height: 165px;
    }
    .crop-rank { font-size: 1.7rem; margin-bottom: 0.2rem; }
    .crop-name { font-size: 1.2rem; font-weight: 750; color: var(--forest-2) !important; }
    .crop-probability { font-size: 0.95rem; color: var(--text-muted) !important; margin: 0.3rem 0 0.55rem 0; }

    .bar-track {
        width: 100%;
        height: 10px;
        background: var(--gray-light);
        border-radius: 999px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--teal), var(--green-light));
        border-radius: 999px;
    }

    /* ---------------- Info cards ---------------- */
    .info-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.3rem;
        border: 1px solid var(--border);
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        min-height: 150px;
    }
    .info-card h3 { color: var(--forest-2) !important; margin-bottom: 0.5rem; font-size: 1.05rem; }
    .info-card p { color: var(--text-dark) !important; line-height: 1.55; font-size: 0.94rem; }

    /* ============================================================
       BUTTONS
       ============================================================ */
    div.stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.65rem 1rem !important;
        width: 100%;
    }
    div.stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--forest-2), var(--green)) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 6px 16px rgba(22, 101, 52, 0.25);
    }
    div.stButton > button[kind="primary"] p,
    button[data-testid="stBaseButton-primary"] p,
    div.stButton > button[kind="primary"] *,
    button[data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
    }
    div.stButton > button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"] {
        background: #ffffff !important;
        border: 1.5px solid var(--forest-2) !important;
        color: var(--forest-2) !important;
    }
    div.stButton > button[kind="secondary"] p,
    button[data-testid="stBaseButton-secondary"] p,
    div.stButton > button[kind="secondary"] *,
    button[data-testid="stBaseButton-secondary"] * {
        color: var(--forest-2) !important;
    }
    div.stButton > button:hover { filter: brightness(1.05); }

    /* ---------------- Download button (separate container from st.button) ---------------- */
    div[data-testid="stDownloadButton"] button {
        background: #ffffff !important;
        border: 1.5px solid var(--forest-2) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        width: 100%;
    }
    div[data-testid="stDownloadButton"] button p,
    div[data-testid="stDownloadButton"] button span,
    div[data-testid="stDownloadButton"] button * {
        color: var(--forest-2) !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: #f0fdf4 !important;
    }

    /* ============================================================
       NUMBER INPUTS — force a light, readable field regardless of
       the surrounding Streamlit theme (light or dark base).
       ============================================================ */
    div[data-testid="stNumberInputContainer"] {
        background: #ffffff !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }
    div[data-testid="stNumberInputContainer"]:focus-within {
        border-color: var(--green) !important;
        box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.15) !important;
    }
    input[data-testid="stNumberInputField"] {
        background: #ffffff !important;
        color: var(--text-dark) !important;
        -webkit-text-fill-color: var(--text-dark) !important;
        font-weight: 600 !important;
    }
    button[data-testid="stNumberInputStepUp"],
    button[data-testid="stNumberInputStepDown"] {
        background: #ffffff !important;
        border-left: 1px solid var(--border) !important;
    }
    button[data-testid="stNumberInputStepUp"] svg,
    button[data-testid="stNumberInputStepDown"] svg {
        fill: var(--forest-2) !important;
        color: var(--forest-2) !important;
    }

    /* Widget labels & help text (applies to all widgets, not just number inputs) */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTooltipIcon"], div[data-testid="stTooltipIcon"] svg {
        color: var(--text-muted) !important;
        fill: var(--text-muted) !important;
    }

    /* ============================================================
       METRICS (used in sidebar + quick stats)
       ============================================================ */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        padding: 0.85rem !important;
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--text-muted) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: var(--forest-2) !important;
        font-weight: 800 !important;
    }

    /* ============================================================
       SIDEBAR — light background, dark readable text throughout.
       ============================================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f7f1 100%) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-dark) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--forest) !important;
        font-weight: 800 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] p {
        color: var(--text-muted) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: var(--forest-2) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stAlertContainer"] {
        background: #eaf6ec !important;
        border: 1px solid #bfe6c4 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stAlertContainer"] * {
        color: var(--forest) !important;
    }

    /* ============================================================
       EXPANDERS
       ============================================================ */
    div[data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        overflow: hidden;
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span {
        color: var(--text-dark) !important;
        font-weight: 700 !important;
    }
    div[data-testid="stExpanderDetails"] {
        background: #ffffff !important;
    }
    div[data-testid="stExpanderDetails"] * {
        color: var(--text-dark) !important;
    }
    div[data-testid="stExpanderDetails"] h3,
    div[data-testid="stExpanderDetails"] h4 {
        color: var(--forest-2) !important;
    }

    /* ============================================================
       ALERTS (info / warning / error), CAPTIONS, DATAFRAMES
       ============================================================ */
    div[data-testid="stAlertContainer"] * {
        color: var(--text-dark) !important;
    }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: var(--text-muted) !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* ---------------- Footer ---------------- */
    .footer {
        text-align: center;
        color: var(--text-muted) !important;
        padding-top: 1.6rem;
        font-size: 0.9rem;
        line-height: 1.7;
    }
    .footer * { color: var(--text-muted) !important; }
    .footer strong { color: var(--forest) !important; }

    hr { border-color: var(--border) !important; }

    /* ============================================================
       RESPONSIVE
       ============================================================ */
    @media (max-width: 768px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero { padding: 1.6rem 1.3rem; border-radius: 18px; }
        .hero-title { font-size: 1.6rem; }
        .result-crop { font-size: 2.2rem; }
        .result-card { padding: 1.5rem; }
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        div[data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
            min-width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# MODEL LOADING

@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    """Load the trained model and its expected feature order.

    Returns (model, feature_names, error_message). error_message is
    None on success; otherwise model/feature_names are None.
    """
    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
    except FileNotFoundError:
        return None, None, (
            "Model files were not found. Please make sure the following "
            f"files are in the same folder as app.py:\n\n"
            f"1. {MODEL_PATH}\n2. {FEATURE_NAMES_PATH}"
        )
    except Exception:
        return None, None, (
            "The model files could not be loaded. They may be corrupted "
            "or saved with an incompatible library version."
        )

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        return None, None, "The loaded model object does not look like a valid classifier."

    feature_names = list(feature_names)
    missing_defaults = [f for f in feature_names if f not in DEFAULT_VALUES]
    if missing_defaults:
        return None, None, (
            "The saved feature list does not match the features this app "
            f"expects. Unknown feature(s): {', '.join(missing_defaults)}"
        )

    return model, feature_names, None


model, feature_names, load_error = load_model()

if load_error:
    st.error(f"❌ {load_error}")
    with st.expander("🔧 Technical details"):
        st.code(f"Expected model file: {MODEL_PATH}\nExpected feature file: {FEATURE_NAMES_PATH}")
    st.stop()



# SESSION STATE INITIALIZATION

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

for _feature, _default in DEFAULT_VALUES.items():
    _key = f"{INPUT_KEY_PREFIX}{_feature}"
    if _key not in st.session_state:
        st.session_state[_key] = _default


def reset_inputs():
    """Restore all input widgets to their default sample values."""
    for feature, default in DEFAULT_VALUES.items():
        st.session_state[f"{INPUT_KEY_PREFIX}{feature}"] = default


def clear_history():
    st.session_state.prediction_history = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌱 Crop AI")
    st.write("AI-based crop recommendation using soil and environmental measurements.")

    st.divider()
    st.markdown("### 🧠 Model")
    st.write(f"**{MODEL_INFO['algorithm']}**")
    st.write(f"**{MODEL_INFO['n_classes']} crop classes**")
    st.write(f"**{MODEL_INFO['n_features']} input features**")

    st.divider()
    st.markdown("### 📊 Model Performance")
    st.metric("Test Accuracy", f"{MODEL_INFO['test_accuracy']:.2f}%")
    st.metric(
        "Correct Predictions",
        f"{MODEL_INFO['correct_predictions']} / {MODEL_INFO['test_samples']}",
    )
    st.metric("Cross-Validation", f"{MODEL_INFO['cv_accuracy']:.2f}%")

    st.divider()
    st.info(
        "This application is a decision-support tool. "
        "Always consider local agricultural conditions before cultivation."
    )


# HERO / HEADER

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">AgriTech · Decision Support</div>
        <div class="hero-title">🌱 AI-Based Crop Recommendation System</div>
        <div class="hero-subtitle">
            Enter soil and environmental measurements to get a data-driven
            crop suggestion from a trained machine-learning model, along
            with confidence scores and practical considerations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# QUICK STATISTICS

stat1, stat2, stat3, stat4 = st.columns(4)
_quick_stats = [
    (f"{MODEL_INFO['test_accuracy']:.2f}%", "Test Accuracy"),
    (f"{MODEL_INFO['n_classes']}", "Crop Classes"),
    (f"{MODEL_INFO['n_features']}", "Input Features"),
    (f"{MODEL_INFO['correct_predictions']}/{MODEL_INFO['test_samples']}", "Correct Predictions"),
]
for _col, (_value, _label) in zip((stat1, stat2, stat3, stat4), _quick_stats):
    with _col:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{_value}</div>
                <div class="stat-label">{_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )



# INPUT SECTION

st.markdown('<div class="section-title">🧪 Soil &amp; Environmental Information</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Enter the measured values from your soil test or environmental data.</div>',
    unsafe_allow_html=True,
)

nutrients_col, chemistry_col, environment_col = st.columns(3)

with nutrients_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="group-label">Soil Nutrients</div>', unsafe_allow_html=True)
    st.number_input(
        "Nitrogen (N)",
        min_value=0.0, max_value=140.0, step=1.0,
        key="input_N", help="Nitrogen concentration in the soil.",
    )
    st.number_input(
        "Phosphorus (P)",
        min_value=0.0, max_value=145.0, step=1.0,
        key="input_P", help="Phosphorus concentration in the soil.",
    )
    st.number_input(
        "Potassium (K)",
        min_value=0.0, max_value=205.0, step=1.0,
        key="input_K", help="Potassium concentration in the soil.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with chemistry_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="group-label">Soil Chemistry</div>', unsafe_allow_html=True)
    st.number_input(
        "Soil pH",
        min_value=0.0, max_value=14.0, step=0.01,
        key="input_ph", help="Soil acidity or alkalinity.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with environment_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="group-label">Environmental Conditions</div>', unsafe_allow_html=True)
    st.number_input(
        "Temperature (°C)",
        min_value=-10.0, max_value=60.0, step=0.1,
        key="input_temperature", help="Environmental temperature.",
    )
    st.number_input(
        "Humidity (%)",
        min_value=0.0, max_value=100.0, step=0.1,
        key="input_humidity", help="Environmental humidity percentage.",
    )
    st.number_input(
        "Rainfall (mm)",
        min_value=0.0, max_value=500.0, step=0.1,
        key="input_rainfall", help="Rainfall amount in millimeters.",
    )
    st.markdown("</div>", unsafe_allow_html=True)



# ACTION BUTTONS

button_col1, button_col2 = st.columns([3, 1])
with button_col1:
    recommend_clicked = st.button(
        "🌾 Analyze & Recommend Crop", use_container_width=True, type="primary"
    )
with button_col2:
    st.button("🔄 Reset Inputs", use_container_width=True, on_click=reset_inputs)


# HELPERS

def collect_inputs() -> dict:
    return {feature: st.session_state[f"{INPUT_KEY_PREFIX}{feature}"] for feature in DEFAULT_VALUES}


def validate_inputs(values: dict) -> list:
    """Return a list of human-readable out-of-range messages (empty if all OK)."""
    problems = []
    for feature, value in values.items():
        minimum, maximum = DATASET_RANGES[feature]
        if value < minimum or value > maximum:
            label = FEATURE_META[feature]["label"]
            problems.append(
                f"**{label}**: {value:.2f} is outside the dataset range "
                f"({minimum:.2f} – {maximum:.2f})."
            )
    return problems


def build_model_dataframe(values: dict, expected_features: list) -> pd.DataFrame:
    """Build a single-row dataframe in the exact column order the model expects."""
    frame = pd.DataFrame([values])
    missing = [f for f in expected_features if f not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature(s) for the model: {', '.join(missing)}")
    return frame[expected_features]


def render_progress_bar(percentage: float):
    st.markdown(
        f"""
        <div class="bar-track">
            <div class="bar-fill" style="width:{max(0, min(100, percentage)):.1f}%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_report(recommended_crop, recommended_probability, top_crops, top_probs, values):
    lines = [
        "AI-BASED CROP RECOMMENDATION SYSTEM",
        "====================================",
        "",
        "RECOMMENDED CROP",
        "----------------",
        "",
        f"Crop: {recommended_crop.capitalize()}",
        f"Model Probability: {recommended_probability * 100:.2f}%",
        "",
        "TOP 3 RECOMMENDATIONS",
        "---------------------",
        "",
    ]
    for i, (crop, prob) in enumerate(zip(top_crops, top_probs), start=1):
        lines.append(f"{i}. {str(crop).capitalize()} — {float(prob) * 100:.2f}%")
    lines += [
        "",
        "SOIL & ENVIRONMENTAL INPUTS",
        "---------------------------",
        "",
    ]
    for feature, value in values.items():
        meta = FEATURE_META[feature]
        lines.append(f"{meta['label']}: {value:.2f} {meta['unit']}".strip())
    lines += [
        "",
        "MODEL INFORMATION",
        "-----------------",
        "",
        f"Model: {MODEL_INFO['algorithm']}",
        f"Number of classes: {MODEL_INFO['n_classes']}",
        f"Input features: {MODEL_INFO['n_features']}",
        f"Test samples: {MODEL_INFO['test_samples']}",
        f"Correct predictions: {MODEL_INFO['correct_predictions']}",
        f"Incorrect predictions: {MODEL_INFO['incorrect_predictions']}",
        f"Test accuracy: {MODEL_INFO['test_accuracy']:.2f}%",
        f"Cross-validation accuracy: {MODEL_INFO['cv_accuracy']:.2f}%",
        "",
        "DISCLAIMER",
        "----------",
        "",
        "This recommendation is generated by a machine-learning model and",
        "should be treated as decision-support information. It does not",
        "guarantee crop success. Consider local weather, soil conditions,",
        "water availability, market demand and professional agricultural",
        "advice before making cultivation decisions.",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    return "\n".join(lines)


# PREDICTION

if recommend_clicked:
    input_values = collect_inputs()
    out_of_range = validate_inputs(input_values)

    if out_of_range:
        st.error("⚠️ Some entered values are outside the range represented in the training dataset.")
        for item in out_of_range:
            st.markdown(f"- {item}")
        st.warning("Please verify the measurement before generating a recommendation.")
    else:
        try:
            input_data = build_model_dataframe(input_values, feature_names)

            prediction = model.predict(input_data)
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            ranked_indices = probabilities.argsort()[::-1]
            top_3_indices = ranked_indices[:3]
            top_3_crops = classes[top_3_indices]
            top_3_probabilities = probabilities[top_3_indices]

            recommended_crop = str(top_3_crops[0])
            recommended_probability = float(top_3_probabilities[0])

            st.session_state.prediction_history.append(
                {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Crop": recommended_crop.capitalize(),
                    "Probability (%)": round(recommended_probability * 100, 2),
                }
            )

            # ---------------- Result card ----------------
            st.divider()
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">🌾 Recommended Crop</div>
                    <div class="result-crop">{recommended_crop.capitalize()}</div>
                    <div class="result-confidence">
                        Model probability: <strong>{recommended_probability * 100:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------------- Top 3 ----------------
            st.markdown('<div class="section-title">🏆 Top 3 Recommendations</div>', unsafe_allow_html=True)
            top_cols = st.columns(3)
            ranks = ["🥇", "🥈", "🥉"]

            for i, col in enumerate(top_cols):
                crop_name = str(top_3_crops[i]).capitalize()
                probability = float(top_3_probabilities[i]) * 100
                with col:
                    st.markdown(
                        f"""
                        <div class="crop-card">
                            <div class="crop-rank">{ranks[i]}</div>
                            <div class="crop-name">{crop_name}</div>
                            <div class="crop-probability">{probability:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    render_progress_bar(probability)

            # ---------------- Confidence chart ----------------
            st.markdown('<div class="section-title">📊 Prediction Confidence</div>', unsafe_allow_html=True)
            chart_data = pd.DataFrame(
                {
                    "Crop": [str(c).capitalize() for c in top_3_crops],
                    "Probability (%)": [float(p) * 100 for p in top_3_probabilities],
                }
            ).set_index("Crop")
            st.bar_chart(chart_data, use_container_width=True, color="#16a34a")
            st.caption(
                "These values represent the model's predicted class probabilities. "
                "They are not guarantees of crop success."
            )

            # ---------------- Crop information ----------------
            st.markdown('<div class="section-title">🌾 Recommended Crop Information</div>', unsafe_allow_html=True)
            crop_details = CROP_INFO.get(recommended_crop.lower(), DEFAULT_CROP_INFO)
            info1, info2 = st.columns(2)
            with info1:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h3>{crop_details['emoji']} {recommended_crop.capitalize()}</h3>
                        <p>{crop_details['description']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with info2:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h3>💡 Practical consideration</h3>
                        <p>{crop_details['advice']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ---------------- Input summary ----------------
            st.markdown('<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)
            summary1, summary2 = st.columns(2)
            with summary1:
                st.metric("Nitrogen (N)", f"{input_values['N']:.2f}")
                st.metric("Phosphorus (P)", f"{input_values['P']:.2f}")
                st.metric("Potassium (K)", f"{input_values['K']:.2f}")
                st.metric("Soil pH", f"{input_values['ph']:.2f}")
            with summary2:
                st.metric("Temperature", f"{input_values['temperature']:.2f} °C")
                st.metric("Humidity", f"{input_values['humidity']:.2f} %")
                st.metric("Rainfall", f"{input_values['rainfall']:.2f} mm")

            # ---------------- Exact model input ----------------
            with st.expander("🔍 View exact input sent to model"):
                st.dataframe(input_data, use_container_width=True, hide_index=True)
                st.write("**Feature order:**")
                st.code(str(feature_names))

            # ---------------- Disclaimer ----------------
            st.warning(
                "⚠️ This recommendation is a machine-learning decision-support result. "
                "Before cultivation, also consider local weather, irrigation or water "
                "availability, soil testing, market demand, seasonal conditions and "
                "advice from agricultural experts."
            )

            # ---------------- Download report ----------------
            report = build_report(
                recommended_crop, recommended_probability, top_3_crops, top_3_probabilities, input_values
            )
            st.download_button(
                label="📥 Download Prediction Report",
                data=report,
                file_name="crop_recommendation_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except Exception as exc:
            st.error("❌ An error occurred while generating the recommendation.")
            with st.expander("🔧 Technical details"):
                st.code(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")



# PREDICTION HISTORY

if st.session_state.prediction_history:
    st.divider()
    header_col, clear_col = st.columns([4, 1])
    with header_col:
        st.markdown('<div class="section-title">📈 Prediction History</div>', unsafe_allow_html=True)
    with clear_col:
        st.button("🗑️ Clear History", use_container_width=True, on_click=clear_history)

    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    if len(history_df) >= 2:
        history_chart = history_df[["Probability (%)"]].copy()
        history_chart.index = range(1, len(history_chart) + 1)
        history_chart.index.name = "Prediction #"
        st.line_chart(history_chart, use_container_width=True, color="#0d9488")


# ABOUT MODEL

st.divider()

with st.expander("ℹ️ About This AI Model"):
    st.markdown(
        f"""
        ### 🌱 Model Overview

        This application uses a **{MODEL_INFO['algorithm']}
        Classification model** for crop recommendation.

        ### 📥 Input Features

        The model uses seven features:

        - Nitrogen (N)
        - Phosphorus (P)
        - Potassium (K)
        - Temperature
        - Humidity
        - Soil pH
        - Rainfall

        ### 🌾 Output

        The model predicts one of **{MODEL_INFO['n_classes']} crop classes**.

        ### 📊 Evaluation

        **Test samples:** {MODEL_INFO['test_samples']}

        **Correct predictions:** {MODEL_INFO['correct_predictions']}

        **Incorrect predictions:** {MODEL_INFO['incorrect_predictions']}

        **Test accuracy:** {MODEL_INFO['test_accuracy']:.2f}%

        **Cross-validation accuracy:** {MODEL_INFO['cv_accuracy']:.2f}%

        ### ⚙️ Tuned Parameters

        - `n_estimators = {MODEL_INFO['params']['n_estimators']}`
        - `max_depth = {MODEL_INFO['params']['max_depth']}`
        - `min_samples_leaf = {MODEL_INFO['params']['min_samples_leaf']}`
        - `min_samples_split = {MODEL_INFO['params']['min_samples_split']}`

        The trained model is loaded from the saved `.pkl` file. New farmer
        inputs are predicted using the existing trained model.
        """
    )

with st.expander("🚀 Future Features"):
    st.markdown(
        """
        ### Planned Extensions

        🌦️ **Live Weather Integration** — Use current weather information to improve environmental context.

        📍 **Location-Based Recommendation** — Use the farmer's location to provide region-specific recommendations.

        💧 **Irrigation Recommendation** — Estimate irrigation requirements based on crop and environmental conditions.

        🧪 **Advanced Soil Analysis** — Add additional soil measurements such as organic carbon and electrical conductivity.

        💰 **Market Information** — Include crop price and market-demand information.

        📅 **Season Recommendation** — Recommend crops according to the current agricultural season.

        🗣️ **Multiple Languages** — Provide the application in regional languages for easier farmer use.

        📷 **Plant Disease Detection** — Add image-based plant disease detection.

        🤖 **AI Agricultural Assistant** — Add a conversational assistant for crop-related questions.

        📱 **Mobile Application** — Convert the recommendation system into a mobile-friendly farmer application.
        """
    )


# FOOTER

st.markdown(
    """
    <div class="footer">
        🌱 <strong>AI-Based Crop Recommendation System</strong>
        <br><br>
        Machine Learning • Agriculture • Decision Support
        <br><br>
        Built with Python • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)