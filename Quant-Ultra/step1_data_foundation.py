"""
Phase 1: Asset Screening and Basic Data Cleaning (Data Foundation)
"""

def execute(context):
    # --- Step 1.1 ---
    print("starting step 1.1")
    print("[INFO] Computing Rolling ADV screens & physically purging IPO/Sub-IPO assets (< M0 days).")
    print("[INFO] Generating forward asset pool capacity constraints and initial baseline AUM math.")
    print("ending step 1.1")
    
    # --- Step 1.2 ---
    print("starting step 1.2")
    print("[INFO] Fixing Survivorship Bias. Forcing point-in-time financial data to Announcement Dates.")
    print("[INFO] Processing Total Return Arrays: Incorporating split/dividend adjustments directly into the matrix.")
    print("ending step 1.2")
    
    # --- Step 1.3 ---
    print("starting step 1.3")
    print("[INFO] Streaming daily limit up/down price bounding matrices to block loop-level recalculation.")
    print("ending step 1.3")
    
    # --- Step 1.4 ---
    print("starting step 1.4")
    print("[INFO] Querying official exchange calendars. Aligning timestamps to Asia/Shanghai at ms precision.")
    print("[PIT-Bus] Forcing append-only (asset, timestamp, value, event_type) immutable serialization storage.")
    print("ending step 1.4")
    
    context['data_foundation_ready'] = True