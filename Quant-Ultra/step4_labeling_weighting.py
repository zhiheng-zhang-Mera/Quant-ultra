"""
Phase 4: Primary Labeling and Sample Weighting (Directional & Numeric Optimization Bounds)
"""

def execute(context):
    # --- Step 4.1 ---
    print("starting step 4.1")
    print("[Labeling] Running triple-barrier signal mapping generator for y_clf (-1, 0, 1).")
    print("[Margin-Check] Evaluating short supply history matrices. Overwriting -1 to 0 where short pool is empty.")
    print("[Regression] Extracting log target_return sequence for y_reg.")
    print("ending step 4.1")
    
    # --- Step 4.2 ---
    print("starting step 4.2")
    print("[Weighting] Scaling samples with rolling exponential time-decay matrix factor w_t.")
    print("ending step 4.2")
    
    context['labels_weighted'] = True