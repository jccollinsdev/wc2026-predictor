"""Team -> confederation map, to let the model learn cross-confederation strength differences
(the root cause of the USA-vs-Paraguay miss: Elo over-rates CONMEBOL schedules, under-rates CONCACAF).

Covers all 48 WC2026 teams plus the major footballing nations. Unknown teams -> 'OTHER'.
"""

UEFA = {
    "Spain", "France", "England", "Germany", "Portugal", "Netherlands", "Belgium", "Italy",
    "Croatia", "Switzerland", "Austria", "Norway", "Sweden", "Denmark", "Scotland", "Poland",
    "Czech Republic", "Turkey", "Serbia", "Ukraine", "Wales", "Hungary", "Romania", "Greece",
    "Russia", "Republic of Ireland", "Northern Ireland", "Iceland", "Finland", "Slovakia",
    "Slovenia", "Bosnia and Herzegovina", "Albania", "North Macedonia", "Georgia", "Bulgaria",
    "Montenegro", "Kosovo", "Israel", "Luxembourg", "Cyprus", "Armenia", "Azerbaijan", "Belarus",
    "Estonia", "Latvia", "Lithuania", "Kazakhstan", "Faroe Islands", "Malta", "Moldova", "Andorra",
    "Gibraltar", "San Marino", "Liechtenstein",
}
CONMEBOL = {
    "Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Peru", "Ecuador", "Paraguay",
    "Bolivia", "Venezuela",
}
CONCACAF = {
    "Mexico", "United States", "Canada", "Costa Rica", "Panama", "Honduras", "Jamaica", "Haiti",
    "El Salvador", "Curaçao", "Trinidad and Tobago", "Guatemala", "Nicaragua", "Suriname",
    "Cuba", "Martinique", "Guadeloupe", "Saint Kitts and Nevis", "Bermuda", "Puerto Rico",
    "Dominican Republic", "Grenada",
}
CAF = {
    "Morocco", "Senegal", "Egypt", "Ivory Coast", "Ghana", "Cape Verde", "South Africa", "Algeria",
    "Tunisia", "DR Congo", "Nigeria", "Cameroon", "Mali", "Burkina Faso", "Guinea", "Zambia",
    "Angola", "Gabon", "Benin", "Uganda", "Equatorial Guinea", "Madagascar", "Mauritania",
    "Namibia", "Mozambique", "Togo", "Sierra Leone", "Liberia", "Congo", "Kenya", "Tanzania",
    "Sudan", "Zimbabwe", "Comoros", "Libya", "Gambia", "Niger", "Ethiopia", "Botswana", "Malawi",
}
AFC = {
    "Japan", "South Korea", "Iran", "Saudi Arabia", "Australia", "Qatar", "Iraq", "Jordan",
    "Uzbekistan", "United Arab Emirates", "China PR", "Oman", "Bahrain", "Syria", "Lebanon",
    "Vietnam", "Thailand", "India", "Indonesia", "Palestine", "Kuwait", "North Korea",
    "Kyrgyzstan", "Tajikistan", "Turkmenistan", "Malaysia", "Philippines", "Afghanistan",
    "Pakistan", "Yemen", "Hong Kong", "Maldives", "Myanmar",
}
OFC = {"New Zealand", "Fiji", "New Caledonia", "Tahiti", "Papua New Guinea", "Solomon Islands",
       "Vanuatu", "Samoa", "Tonga"}

_MAP = {}
for conf, members in [("UEFA", UEFA), ("CONMEBOL", CONMEBOL), ("CONCACAF", CONCACAF),
                      ("CAF", CAF), ("AFC", AFC), ("OFC", OFC)]:
    for t in members:
        _MAP[t] = conf

# ordinal encoding for tree models (also keep string for diagnostics)
CONF_CODE = {"UEFA": 0, "CONMEBOL": 1, "CONCACAF": 2, "CAF": 3, "AFC": 4, "OFC": 5, "OTHER": 6}


def confederation(team):
    return _MAP.get(team, "OTHER")


def conf_code(team):
    return CONF_CODE[confederation(team)]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from tournament import GROUPS
    from collections import Counter
    wc = [t for ts in GROUPS.values() for t in ts]
    c = Counter(confederation(t) for t in wc)
    print("WC2026 teams by confederation:", dict(c))
    unknown = [t for t in wc if confederation(t) == "OTHER"]
    print("unmapped WC teams (should be none):", unknown)
