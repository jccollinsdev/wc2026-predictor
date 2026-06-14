"""Refresh the core match data from public, no-auth mirrors (no API key needed).

Run this to pull the latest international results/goals/shootouts (e.g. as the World Cup progresses,
so new results settle predictions in the tracker):

    python scripts/fetch_data.py

The large optional files (players_multiyear.csv, fixtures.csv, match_stats.csv) are gitignored; see
the README for their sources. The model runs on current-edition player ratings (players.csv) without
them.
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(HERE), "data", "raw")

# martj42/international_results — CC0, the canonical international match dataset
BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = {
    "results.csv": f"{BASE}/results.csv",
    "goalscorers.csv": f"{BASE}/goalscorers.csv",
    "shootouts.csv": f"{BASE}/shootouts.csv",
}


def main():
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(RAW, name)
        try:
            print(f"fetching {name} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            print(f"ok ({os.path.getsize(dest)//1024} KB)")
        except Exception as e:
            print(f"FAILED: {e}")
    print("done. (player ratings: see README for EAFC26 / multi-year FIFA sources)")


if __name__ == "__main__":
    main()
