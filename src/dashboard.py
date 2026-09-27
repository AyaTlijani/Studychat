from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = Path("results")
PROFILE_FILE = RESULTS_DIR / "student_profiles.csv"
LOGO_PATH = Path(r"C:\Users\MSI\Desktop\I5-project-esprim\src\studycchatlogo.png")
# Fallback relative path if you run the app from the project root:
# LOGO_PATH = Path("src/studycchatlogo.png")

st.set_page_config(
    page_title="StudyChat | Candidate Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# COMPETENCY DEFINITIONS
# ============================================================

COMPETENCIES = {
    "Autonomy": {
        "percentile": "autonomy_percentile",
        "level": "autonomy_level",
        "rate": "autonomy_evidence_rate",
        "count": "autonomy_evidence_count",
        "interactions": "autonomy_interactions",
        "description": (
            "Evidence that the student actively participates in the work "
            "instead of simply delegating the task to the assistant."
        ),
        "positive": [
            "Requests explanations or conceptual understanding",
            "Provides context about the work or problem",
            "Edits or refines generated material",
        ],
        "negative": [
            "Direct delegation of the task to the assistant",
        ],
    },
    "Diagnostic": {
        "percentile": "diagnostic_percentile",
        "level": "diagnostic_level",
        "rate": "diagnostic_evidence_rate",
        "count": "diagnostic_evidence_count",
        "interactions": "diagnostic_interactions",
        "description": (
            "Evidence that the student identifies, investigates and works "
            "through problems."
        ),
        "positive": [
            "Uses diagnostic/problem-description language",
            "Provides relevant context for a problem",
            "Edits or modifies a solution",
            "Checks or verifies a result",
        ],
        "negative": [],
    },
    "Critical Thinking": {
        "percentile": "critical_percentile",
        "level": "critical_level",
        "rate": "critical_evidence_rate",
        "count": "critical_evidence_count",
        "interactions": "critical_interactions",
        "description": (
            "Evidence that the student checks, questions or evaluates "
            "information rather than simply accepting an answer."
        ),
        "positive": [
            "Explicitly verifies an answer or result",
            "Requests explanation or justification",
        ],
        "negative": [],
    },
}


# ============================================================
# LIGHT PROFESSIONAL THEME + SOFT YELLOW ACCENTS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #f8f9fc;
        color: #172033;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e8e4d9;
    }

    [data-testid="stSidebar"] * {
        color: #172033 !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #334155 !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 5px;
    }

    /* ========================================================
       HEADINGS
       ======================================================== */

    h1 {
        color: #172033 !important;
        font-size: 2.15rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.6px;
        margin-bottom: 0.2rem !important;
    }

    h2 {
        color: #172033 !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #26344d !important;
        font-weight: 700 !important;
    }

    h4 {
        color: #334155 !important;
        font-weight: 650 !important;
    }

    /* ========================================================
       MUTED TEXT
       ======================================================== */

    .muted {
        color: #64748b !important;
        font-size: 0.9rem;
    }

    [data-testid="stCaptionContainer"] {
        color: #64748b !important;
    }

    [data-testid="stMarkdownContainer"] {
        color: #334155;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #334155 !important;
    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e8e4d9 !important;
        border-radius: 14px !important;
        padding: 18px 20px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-weight: 650 !important;
    }

    [data-testid="stMetricValue"] {
        color: #172033 !important;
        font-size: 1.7rem !important;
        font-weight: 750 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #64748b !important;
    }

    /* ========================================================
       SELECT BOX
       ======================================================== */

    .stSelectbox label {
        color: #334155 !important;
        font-weight: 650 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #e0d9c8 !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="select"] span {
        color: #172033 !important;
    }

    /* Dropdown */
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }

    div[data-baseweb="popover"] * {
        color: #172033 !important;
    }

    /* ========================================================
       PROGRESS  (soft yellow accent)
       ======================================================== */

    .stProgress > div > div {
        background-color: #f3efe6 !important;
        border-radius: 20px !important;
    }

    .stProgress > div > div > div {
        background-color: #e8b923 !important;   /* soft logo yellow */
        border-radius: 20px !important;
    }

    /* ========================================================
       ALERTS
       ======================================================== */

    .stAlert {
        border-radius: 10px !important;
    }

    .stAlert p {
        color: #334155 !important;
    }

    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #e8e4d9 !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #e8e4d9 !important;
        margin-top: 1.7rem !important;
        margin-bottom: 1.7rem !important;
    }

    /* ========================================================
       CODE
       ======================================================== */

    code {
        color: #334155 !important;
    }

    /* ========================================================
       RADIO BUTTONS  (yellow accent)
       ======================================================== */

    [data-testid="stSidebar"] input[type="radio"] {
        accent-color: #e8b923;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

if not PROFILE_FILE.exists():
    st.error(
        "student_profiles.csv was not found.\n\n"
        "Run `python src/analysis.py` first."
    )
    st.stop()

profiles = pd.read_csv(PROFILE_FILE)

if profiles.empty:
    st.error("student_profiles.csv is empty.")
    st.stop()


# ============================================================
# SIDEBAR  (with logo)
# ============================================================

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)
else:
    st.sidebar.markdown("## StudyChat")

st.sidebar.caption("Candidate Analytics")
st.sidebar.divider()
st.sidebar.markdown("**VIEW**")

page = st.sidebar.radio(
    "",
    [
        "Candidate Dashboard",
        "How It Works",
    ],
    label_visibility="collapsed",
)


# ============================================================
# HOW IT WORKS
# ============================================================

if page == "How It Works":

    st.title("How the analysis works")

    st.markdown(
        '<p class="muted">'
        "From StudyChat interaction traces to observable competency evidence"
        "</p>",
        unsafe_allow_html=True,
    )

    st.info(
        "The system analyzes observable evidence in StudyChat interactions. "
        "It does not claim to measure a student's true competency, "
        "personality, or future job performance."
    )

    st.divider()

    st.subheader("1. Interaction data")

    st.write(
        "The analysis starts from the student's StudyChat interactions, "
        "including prompts, interaction labels and behavioral indicators."
    )

    st.subheader("2. Competency-specific evidence")

    st.write(
        "Each competency has its own evidence definition. "
        "The same formula is not applied to all competencies."
    )

    for competency, info in COMPETENCIES.items():

        st.markdown(f"#### {competency}")

        st.write(info["description"])

        st.markdown("**Evidence considered:**")

        for item in info["positive"]:
            st.write(f"• {item}")

        if info["negative"]:

            st.markdown("**Negative evidence:**")

            for item in info["negative"]:
                st.write(f"• {item}")

    st.divider()

    st.subheader("3. Evidence rate")

    st.write(
        "For each student and competency, we count the interactions "
        "that contain observable evidence."
    )

    st.code(
        "Evidence rate = evidence-bearing interactions / analyzed interactions"
    )

    st.info(
        "The evidence rate is not interpreted as 'percentage of competency'. "
        "It describes how frequently observable evidence appears in the "
        "student's analyzed interactions."
    )

    st.subheader("4. Cohort percentile")

    st.write(
        "Students are compared with the other students in the StudyChat "
        "cohort using their evidence rate."
    )

    st.code(
        "Percentile = relative position within the StudyChat cohort"
    )

    st.write(
        "For example, the 80th percentile means the student's observed "
        "evidence rate is higher than approximately 80% of the cohort."
    )

    st.subheader("5. Interpretation")

    interpretation = pd.DataFrame(
        {
            "Cohort percentile": [
                "75–100",
                "25–75",
                "0–25",
            ],
            "Interpretation": [
                "Strong observed evidence",
                "Moderate observed evidence",
                "Developing observed evidence",
            ],
        }
    )

    st.dataframe(
        interpretation,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Important limitation")

    st.warning(
        "The dataset does not provide independent teacher or employer "
        "assessments of these competencies. Therefore, the results should "
        "be treated as automated competency-evidence analysis and "
        "decision support, not as a validated hiring assessment."
    )

    st.stop()


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title("Candidate Dashboard")

st.caption("Observable competency evidence from StudyChat learning traces")


# ============================================================
# OVERVIEW METRICS
# ============================================================

total_candidates = len(profiles)

total_interactions = int(
    profiles["total_interactions"].sum()
)

strong_autonomy = int(
    (profiles["autonomy_level"] == "Strong").sum()
)

strong_diagnostic = int(
    (profiles["diagnostic_level"] == "Strong").sum()
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Candidates",
        f"{total_candidates:,}",
    )

with col2:
    st.metric(
        "Interactions",
        f"{total_interactions:,}",
    )

with col3:
    st.metric(
        "Strong Autonomy",
        strong_autonomy,
    )

with col4:
    st.metric(
        "Strong Diagnostic",
        strong_diagnostic,
    )


# ============================================================
# CANDIDATE SELECTION
# ============================================================

st.divider()

st.subheader("Candidate")

candidate_ids = (
    profiles["userId"]
    .astype(str)
    .tolist()
)

selected_id = st.selectbox(
    "Select a candidate",
    candidate_ids,
)

candidate = profiles[
    profiles["userId"].astype(str) == selected_id
].iloc[0]


# ============================================================
# CANDIDATE QUICK SUMMARY
# ============================================================

q1, q2, q3 = st.columns(3)

with q1:
    st.metric(
        "Cohort rank",
        f"#{int(candidate['cohort_rank'])}",
    )

with q2:
    st.metric(
        "Interactions analyzed",
        f"{int(candidate['total_interactions']):,}",
    )

with q3:
    st.metric(
        "Evidence observations",
        f"{int(candidate['total_evidence']):,}",
    )


# ============================================================
# COMPETENCY PROFILE
# ============================================================

st.divider()

st.subheader("Competency Profile")

st.caption("Relative position of the candidate within the StudyChat cohort.")

col1, col2, col3 = st.columns(3)

for column, competency in zip(
    [col1, col2, col3],
    [
        "Autonomy",
        "Diagnostic",
        "Critical Thinking",
    ],
):

    info = COMPETENCIES[competency]

    percentile_value = float(
        candidate[info["percentile"]]
    )

    level = str(
        candidate[info["level"]]
    )

    with column:

        st.markdown(f"### {competency}")

        st.metric(
            "Percentile",
            f"{percentile_value:.0f}",
        )

        st.progress(
            min(max(percentile_value / 100, 0), 1)
        )

        if level == "Strong":

            st.success(
                "Strong observed evidence"
            )

        elif level == "Moderate":

            st.warning(
                "Moderate observed evidence"
            )

        else:

            st.info(
                "Developing observed evidence"
            )


# ============================================================
# PROFILE SUMMARY + COMPETENCY DETAILS
# ============================================================

st.divider()

left, right = st.columns(2)

with left:

    st.subheader("Profile Summary")

    st.write(
        f"**Strongest observed area:** "
        f"{candidate['strongest_competency']}"
    )

    st.write(
        f"**Lowest observed area:** "
        f"{candidate['development_area']}"
    )

    st.write(
        f"**Cohort position:** "
        f"#{int(candidate['cohort_rank'])} "
        f"of {total_candidates}"
    )

    st.write(
        f"**Interactions analyzed:** "
        f"{int(candidate['total_interactions']):,}"
    )

    st.write(
        f"**Evidence observations:** "
        f"{int(candidate['total_evidence']):,}"
    )


with right:

    st.subheader("Competency Details")

    selected_competency = st.selectbox(
        "Inspect competency",
        list(COMPETENCIES.keys()),
        key="competency_selector",
    )

    info = COMPETENCIES[selected_competency]

    percentile_value = float(
        candidate[info["percentile"]]
    )

    evidence_count = int(
        candidate[info["count"]]
    )

    interactions = int(
        candidate[info["interactions"]]
    )

    evidence_rate = float(
        candidate[info["rate"]]
    )

    level = str(
        candidate[info["level"]]
    )

    st.write(
        info["description"]
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Percentile",
            f"{percentile_value:.0f}",
        )

    with d2:
        st.metric(
            "Evidence",
            evidence_count,
        )

    with d3:
        st.metric(
            "Interactions",
            interactions,
        )

    st.caption(
        f"Evidence rate: {evidence_rate:.1%} · "
        f"{level} observed evidence"
    )


# ============================================================
# EVIDENCE
# ============================================================

st.divider()

st.subheader(
    f"Evidence used for {selected_competency}"
)

evidence_col1, evidence_col2 = st.columns(2)

with evidence_col1:

    st.markdown("**Positive evidence**")

    for item in info["positive"]:
        st.write(f"✓ {item}")


with evidence_col2:

    st.markdown("**Negative evidence**")

    if info["negative"]:

        for item in info["negative"]:
            st.write(f"− {item}")

    else:

        st.caption(
            "No negative evidence is currently defined."
        )


# ============================================================
# CANDIDATE COMPARISON
# ============================================================

st.divider()

st.subheader("Candidate Comparison")

st.caption("Relative position of candidates within the StudyChat cohort.")

comparison = profiles[
    [
        "userId",
        "autonomy_percentile",
        "diagnostic_percentile",
        "critical_percentile",
        "cohort_rank",
        "strongest_competency",
        "development_area",
    ]
].copy()

comparison.columns = [
    "Candidate",
    "Autonomy",
    "Diagnostic",
    "Critical Thinking",
    "Rank",
    "Strongest",
    "Lowest observed area",
]

comparison = comparison.sort_values(
    "Rank",
    ascending=True,
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Candidate": st.column_config.TextColumn(
            "Candidate",
            width="medium",
        ),
        "Autonomy": st.column_config.NumberColumn(
            "Autonomy percentile",
            format="%.0f",
        ),
        "Diagnostic": st.column_config.NumberColumn(
            "Diagnostic percentile",
            format="%.0f",
        ),
        "Critical Thinking": st.column_config.NumberColumn(
            "Critical Thinking percentile",
            format="%.0f",
        ),
        "Rank": st.column_config.NumberColumn(
            "Cohort rank",
            format="%d",
        ),
        "Strongest": st.column_config.TextColumn(
            "Strongest",
            width="medium",
        ),
        "Lowest observed area": st.column_config.TextColumn(
            "Lowest observed area",
            width="medium",
        ),
    },
)

st.caption(
    "Percentile values indicate relative position within the StudyChat "
    "cohort. They are not percentages of competency."
)