"""Injury/availability overlay for WC2026 (sourced from ESPN's injury tracker, June 2026).

Players ruled OUT are removed from their nation's EA FC 26 player pool before squad-strength is
aggregated, so teams missing key players are correctly downgraded at prediction time. Applied only
to live predictions (edition 26) — the model is trained on clean strength and applies that learned
mapping to the actually-available squad.
"""

# (player name as it appears in EA FC data short/long name, nation as nationality_name)
INJURED_OUT = [
    ("Rodrygo", "Brazil"), ("Estêvão", "Brazil"), ("Éder Militão", "Brazil"),
    ("Gnabry", "Germany"), ("Lennart Karl", "Germany"), ("ter Stegen", "Germany"),
    ("Timber", "Netherlands"), ("Xavi Simons", "Netherlands"), ("Schouten", "Netherlands"),
    ("de Ligt", "Netherlands"),
    ("Mitoma", "Japan"), ("Minamino", "Japan"),
    ("Ekitiké", "France"),
    ("Agyemang", "United States"), ("Cardoso", "United States"),
    ("Miller", "Australia"),
    ("Fermín", "Spain"),
    ("Flores", "Canada"),
    ("Gilmour", "Scotland"),
]


def injured_by_nation():
    d = {}
    for player, nation in INJURED_OUT:
        d.setdefault(nation, []).append(player)
    return d
