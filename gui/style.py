"""Visual style for the PEMFC Simulator GUI.

Layered in three pieces:

* ``apply_matplotlib()`` -- research-paper rcParams for the embedded
  figures: Okabe-Ito categorical palette, cividis sequential cmap,
  serif fonts, light grid, clean spines. Safe to call at module import.
* ``apply_streamlit()`` -- CSS injection that gives the GUI a clean
  scientific-software / academic-publishing look: cool off-white page,
  white cards with a 1.5 px slate-blue outline and soft shadow,
  navy accent, section headers with an underlined accent rule, and
  monospace tabular numbers for the parameter inputs. Must be called
  inside the run context, AFTER ``st.set_page_config(...)``.

Design references drawn from Nature.com, IEEE Xplore, COMSOL Desktop,
OriginPro, and the Tailwind "Slate"/"Navy" academic palette.
"""

import matplotlib as mpl
import streamlit as st


# Okabe-Ito 8-colour palette: colour-blind safe, recommended by Nature
# and widely used in IEEE / journal figures.
PALETTE = [
    "#1e3a5f",  # deep navy (matches the GUI accent)
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#000000",  # black
]


# ---------------------------------------------------------------------------
# Streamlit chrome -- scientific-website inspired light theme.
# ---------------------------------------------------------------------------
_STREAMLIT_CSS = """
<style>
/* ===========================================================================
   Hero title block (centered Playfair Display + Inter caps subtitle)
   The structure injected from gui/app.py is:
     <div class="pemfc-hero">
       <div class="title">PEMFC <span class="accent">Simulator</span></div>
       <div class="rule"></div>
       <div class="subtitle">…</div>
     </div>
   =========================================================================== */
.pemfc-hero {
    text-align: center;
    margin: 0.4rem 0 1.1rem 0;
    padding: 0;
}
.pemfc-hero .title {
    font-family: "Playfair Display", "Cormorant Garamond", Georgia, serif !important;
    font-size: 2.85rem !important;
    font-weight: 700 !important;
    color: #0f172a;
    letter-spacing: -0.018em;
    line-height: 1.05;
    margin: 0;
}
.pemfc-hero .title .accent {
    color: #1e3a5f;
    font-style: italic;
    font-weight: 600;
}
.pemfc-hero .rule {
    width: 56px;
    height: 2px;
    background: #1e3a5f;
    margin: 0.55rem auto 0.55rem auto;
    border-radius: 2px;
}
.pemfc-hero .subtitle {
    font-family: "Inter", -apple-system, "Segoe UI", Arial, sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500;
    color: #64748b;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}

/* ===========================================================================
   Design tokens
   =========================================================================== */
:root {
    --bg-page:        #f4f6fa;     /* cool off-white page background  */
    --bg-card:        #ffffff;     /* white panels                    */
    --bg-input:       #ffffff;
    --bg-input-soft:  #f8fafc;     /* slightly recessed inputs        */
    --fg-strong:      #0f172a;     /* primary text (slate-900)        */
    --fg-body:        #1e293b;     /* body text (slate-800)           */
    --fg-mute:        #64748b;     /* captions / hints (slate-500)    */
    --border:         #cbd5e1;     /* card outline (slate-300)        */
    --border-soft:    #e2e8f0;     /* inner dividers                  */
    --accent:         #1e3a5f;     /* deep navy primary accent        */
    --accent-hover:   #2c5282;     /* steel blue on hover             */
    --accent-tint:    #e6edf5;     /* very pale navy for selections   */
    --shadow-sm:      0 1px 2px rgba(15, 23, 42, 0.06);
    --shadow-md:      0 1px 3px rgba(15, 23, 42, 0.08),
                      0 1px 2px rgba(15, 23, 42, 0.04);
    --radius-card:    8px;
    --radius-input:   5px;
}

/* ===========================================================================
   Page background + typography
   =========================================================================== */
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-page) !important;
}
[data-testid="stHeader"] {
    background: linear-gradient(to bottom,
        rgba(244, 246, 250, 0.96),
        rgba(244, 246, 250, 0));
    backdrop-filter: none !important;
}

html, body, [class*="css"] {
    font-family: "Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont,
                 "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
    color: var(--fg-body) !important;
    -webkit-font-smoothing: antialiased;
    font-feature-settings: "ss01", "cv11", "tnum";
}

/* Top app header */
.stMarkdown h3 {
    color: var(--fg-strong) !important;
    font-weight: 700 !important;
    letter-spacing: -0.015em !important;
    font-size: 1.45rem !important;
    margin-bottom: 0.2rem !important;
}

/* ===========================================================================
   Section cards
   ===========================================================================
   IMPORTANT: Streamlit emits ``data-testid="stVerticalBlockBorderWrapper"``
   on *every* vertical block, even non-bordered ones (it controls scroll
   behaviour). Styling that testid would border every block on the page
   and produce visible "double frames". Streamlit's bordered-container
   class for 1.32 is ``st-emotion-cache-z1rf3o`` -- we scope styling to
   that class so only ``st.container(border=True)`` cards get the look.

   ``:has()`` fallback covers future versions where the hash might change:
   any wrapper whose direct child is the inner block AND which has a
   visible border picks up the styling.
   =========================================================================== */
.st-emotion-cache-z1rf3o,
[data-testid="stVerticalBlockBorderWrapper"][data-test-border="true"] {
    background-color: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-card) !important;
    box-shadow: var(--shadow-md) !important;
    padding: 1.35rem 1.5rem 1.25rem 1.5rem !important;
    margin-bottom: 0.25rem;
}

/* Section header "#### § 1 Parameters" -- academic-paper style:
   a navy underline that only spans the title text, not the full card. */
.st-emotion-cache-z1rf3o h4,
.st-emotion-cache-z1rf3o h5,
[data-testid="stVerticalBlockBorderWrapper"][data-test-border="true"] h4,
[data-testid="stVerticalBlockBorderWrapper"][data-test-border="true"] h5 {
    display: inline-block !important;     /* shrink to text width */
    width: auto !important;
    color: var(--fg-strong) !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    letter-spacing: -0.005em !important;
    border-bottom: 2.5px solid var(--accent) !important;
    padding-bottom: 0.35rem !important;
    margin: 0.1rem 0 1rem 0 !important;
    text-transform: none !important;
}
/* Force a line break so following content sits below the inline-block h4 */
.st-emotion-cache-z1rf3o h4::after,
.st-emotion-cache-z1rf3o h5::after,
[data-testid="stVerticalBlockBorderWrapper"][data-test-border="true"] h4::after,
[data-testid="stVerticalBlockBorderWrapper"][data-test-border="true"] h5::after {
    content: "";
    display: block;
}
/* Sub-headers inside sections (e.g. "Time span & solver", "Mesh") */
.st-emotion-cache-z1rf3o .stMarkdown p strong,
[data-testid="stVerticalBlockBorderWrapper"][data-test-border="true"] .stMarkdown p strong {
    color: var(--fg-strong);
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.005em;
}

/* ===========================================================================
   Form inputs -- tabular numbers, slim borders, navy focus ring
   =========================================================================== */
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
    font-family: "JetBrains Mono", "SF Mono", "Consolas",
                 ui-monospace, monospace !important;
    font-variant-numeric: tabular-nums;
    font-size: 0.88rem !important;
    background-color: var(--bg-input) !important;
    color: var(--fg-strong) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-input) !important;
}
.stNumberInput input:focus,
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-tint) !important;
}

/* Selectbox / dropdown surface */
[data-baseweb="select"] > div {
    background-color: var(--bg-input) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-input) !important;
    font-size: 0.9rem !important;
}
[data-baseweb="popover"], [role="listbox"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-md);
}
[role="option"]:hover { background-color: var(--accent-tint) !important; }

/* Labels above inputs -- compact and slate */
label,
.stRadio label,
.stCheckbox label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label {
    color: var(--fg-mute) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.015em;
    text-transform: none;
}

/* ===========================================================================
   Radio buttons (horizontal Model variant + Test profile + Auxiliary system)

   Streamlit renders each option as:
     <label data-baseweb="radio">
       <div>  (the visible bullet — Streamlit hard-codes #ff4b4b)
         <input type="radio">
       </div>
       ...
     </label>
   We override the bullet's fill via :has(input:checked) and pill-shape
   the surrounding label.
   =========================================================================== */
[data-baseweb="radio"]:has(input:checked) > div:first-child {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}
[data-baseweb="radio"]:hover > div:first-child {
    border-color: var(--accent) !important;
}
.stRadio > div[role="radiogroup"] {
    gap: 0.55rem !important;
}
.stRadio > div[role="radiogroup"] > label {
    background-color: var(--bg-input-soft);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.2rem 0.85rem 0.2rem 0.6rem;
    transition: all 0.12s ease;
}
.stRadio > div[role="radiogroup"] > label:hover {
    background-color: var(--accent-tint);
    border-color: var(--accent);
}
/* Selected pill gets a soft navy tint background so the active option is
   obvious even at a glance, not just by the dot colour. */
.stRadio > div[role="radiogroup"] > label:has(input:checked) {
    background-color: var(--accent-tint);
    border-color: var(--accent);
}

/* ===========================================================================
   Multiselect chips -- Streamlit defaults to bright red (#ff4b4b).
   Override to the navy accent so the region filter doesn't clash.
   =========================================================================== */
[data-baseweb="tag"] {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
}
[data-baseweb="tag"] svg { fill: rgba(255, 255, 255, 0.85) !important; }
[data-baseweb="tag"]:hover svg { fill: #ffffff !important; }

/* ===========================================================================
   Buttons
   =========================================================================== */
.stButton > button,
.stDownloadButton > button {
    border-radius: var(--radius-input) !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    border: 1px solid var(--border) !important;
    background-color: var(--bg-card) !important;
    color: var(--fg-body) !important;
    transition: all 0.12s ease;
    box-shadow: var(--shadow-sm);
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background-color: var(--accent-tint) !important;
}
/* Primary button: navy fill, white text */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
    color: #ffffff !important;
}

/* ===========================================================================
   Expanders (parameter groups in Section 1)
   =========================================================================== */
[data-testid="stExpander"], .stExpander {
    background-color: var(--bg-input-soft) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: var(--radius-input) !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary,
.stExpander summary {
    color: var(--fg-strong) !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    text-transform: none !important;
    letter-spacing: 0.005em;
}
[data-testid="stExpander"] summary:hover {
    color: var(--accent) !important;
}

/* ===========================================================================
   Tabs (results section)
   =========================================================================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 1.25rem !important;
    border-bottom: 1px solid var(--border-soft) !important;
    margin-bottom: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--fg-mute) !important;
    font-weight: 500;
    font-size: 0.92rem !important;
    border: none !important;
    padding: 0.4rem 0.1rem !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2.5px solid var(--accent) !important;
}

/* ===========================================================================
   Captions, dividers, alerts
   =========================================================================== */
[data-testid="stCaptionContainer"], .stCaption, small {
    color: var(--fg-mute) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.005em;
}
hr, [data-testid="stMarkdownContainer"] hr {
    border-color: var(--border-soft) !important;
    margin: 0.6rem 0 0.9rem 0 !important;
}
[data-baseweb="notification"], .stAlert > div {
    border-radius: var(--radius-input) !important;
    border: 1px solid var(--border) !important;
}

/* Spinner accent */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ===========================================================================
   Scrollbar inside the section cards -- slim, slate
   =========================================================================== */
[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar {
    width: 7px; height: 7px;
}
[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb {
    background-color: #94a3b8;
    border-radius: 999px;
}
[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb:hover {
    background-color: var(--accent);
}
</style>
"""


def apply_matplotlib():
    """Research-paper rcParams for embedded figures (white background)."""
    mpl.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,

        # Axes & spines
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#1e293b",
        "axes.labelcolor": "#0f172a",
        "axes.titlecolor": "#0f172a",

        # Grid
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.color": "#94a3b8",

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#1e293b",
        "ytick.color": "#1e293b",

        # Lines & markers
        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        # Legend
        "legend.frameon": False,
        "legend.borderaxespad": 0.4,

        # Figure & saved output
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # Colour cycle (lines) and default sequential colormap (heatmaps)
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "image.cmap": "cividis",
    })


def apply_streamlit():
    """Inject the scientific-publication-style CSS."""
    st.markdown(_STREAMLIT_CSS, unsafe_allow_html=True)


def apply():
    """Apply both matplotlib + Streamlit styling. Call once at app start."""
    apply_matplotlib()
    apply_streamlit()
