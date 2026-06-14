"""Global configuration and constants for the WC2026 predictor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
for _p in (DATA_PROC, MODELS, OUTPUTS):
    _p.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = DATA_RAW / "results.csv"
GOALSCORERS_CSV = DATA_RAW / "goalscorers.csv"
SHOOTOUTS_CSV = DATA_RAW / "shootouts.csv"

# --- Dixon-Coles ---
DC_TRAIN_FROM = "2010-01-01"   # window start; time-decay handles recency within it
DC_TIME_DECAY = 0.0012         # xi per day (half-life ~578 days); tuned by walk-forward RPS sweep
DC_FRIENDLY_WEIGHT = 0.70      # down-weight noisy friendlies in the likelihood (tuned)
DC_REG = 0.01                  # L2 on attack/defense (breaks degeneracy + shrinks sparse teams)
DC_MAX_GOALS = 12              # scoreline grid 0..MAX
DC_RHO_BOUNDS = (-0.20, 0.20)

# --- First half ---
# Measured directly from goalscorers.csv (minute<=45): 21297/47606 = 0.4474
FIRST_HALF_FRACTION = 0.4474

# --- Elo (World Football Elo Ratings spec) ---
ELO_INIT = 1500.0
ELO_HOME_ADV = 100.0           # added to home rating in expectation, only when neutral=FALSE

# --- Host nations (play "at home" even though WC fixtures are flagged neutral=TRUE for non-hosts) ---
HOST_NATIONS = {"United States", "Canada", "Mexico"}

# tournament-name -> Elo K (match importance)
def elo_k_for_tournament(t: str) -> float:
    s = (t or "").lower()
    if "world cup" in s and "qual" not in s:
        return 60.0
    if "qual" in s:                                   # WC + continental qualifiers
        return 40.0
    # continental championship finals + major intercontinental
    for key in ("uefa euro", "copa am", "african cup", "afc asian cup",
                "gold cup", "concacaf championship", "confederations cup",
                "nations league finals", "oceania nations cup", "asian cup"):
        if key in s:
            return 50.0
    if "nations league" in s or "confederations" in s:
        return 40.0
    if "friendly" in s:
        return 20.0
    return 30.0                                       # other tournaments

# tournament tier as an ordinal feature (higher = more important/serious)
def tournament_tier(t: str) -> int:
    s = (t or "").lower()
    if "world cup" in s and "qual" not in s:
        return 5
    for key in ("uefa euro", "copa am", "african cup", "afc asian cup",
                "gold cup", "concacaf championship", "asian cup"):
        if key in s:
            return 4
    if "nations league" in s or "confederations" in s:
        return 3
    if "qual" in s:
        return 3
    if "friendly" in s:
        return 0
    return 1

FORM_WINDOW = 5                # rolling form window (matches)
