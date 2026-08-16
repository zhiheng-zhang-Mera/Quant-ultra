"""Tests for the R19.7 cash-flow-quality factor pipeline (PIT fundamentals)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Main.fundamental_factors import FUNDAMENTAL_FIELDS, build_fundamental_panels, load_all_fundamentals  # noqa: E402


def test_ocf_np_registered_in_fundamental_fields():
    assert "ocf_np" in FUNDAMENTAL_FIELDS
    assert FUNDAMENTAL_FIELDS["ocf_np"] == "ocf_np"


def test_load_all_fundamentals_merges_cashflow_cache(tmp_path):
    cash = {
        "600519.SH": [
            {"pub_date": "2026-08-15", "stat_date": "2026-06-30", "ocf_np": 1.54},
            {"pub_date": "2026-04-25", "stat_date": "2026-03-31", "ocf_np": 1.22},
        ]
    }
    (tmp_path / "fundamentals_cashflow.json").write_text(
        json.dumps(cash, ensure_ascii=False), encoding="utf-8")
    merged = load_all_fundamentals(
        profit_path=tmp_path / "missing_profit.json",  # absent -> empty
        balance_path=tmp_path / "missing_balance.json",
        cashflow_path=tmp_path / "fundamentals_cashflow.json",
    )
    assert "600519.SH" in merged
    assert merged["600519.SH"][0]["ocf_np"] == 1.54


def test_ocf_np_panel_is_pit_forward_filled():
    data = {
        "600519.SH": [
            {"pub_date": "2026-08-15", "stat_date": "2026-06-30", "ocf_np": 1.54},
            {"pub_date": "2026-04-25", "stat_date": "2026-03-31", "ocf_np": 1.22},
        ]
    }
    common = pd.date_range("2026-01-01", periods=200, freq="B")
    panel = build_fundamental_panels(data, common, ["600519.SH"], {"ocf_np": "ocf_np"})["ocf_np"]
    s = panel["600519.SH"]
    # before the first announcement date -> NaN (no look-ahead)
    assert pd.isna(s.loc[pd.Timestamp("2026-01-05")])
    # after the Q1 announcement -> Q1 value, unchanged until the H1 announcement
    assert np.isclose(s.loc[pd.Timestamp("2026-05-04")], 1.22)
    assert np.isclose(s.loc[pd.Timestamp("2026-08-12")], 1.22)
    # on/after the H1 announcement -> H1 value
    if pd.Timestamp("2026-08-17") in s.index:
        assert np.isclose(s.loc[pd.Timestamp("2026-08-17")], 1.54)
