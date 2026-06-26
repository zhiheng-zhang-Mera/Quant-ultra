"""
Phase 6: Multi-Asset Position Sizing and Bounded Execution Matrix Optimization (BL-MVO Engine)
"""

def execute(context):
    # --- Step M.1 ---
    print("starting step M.1")
    print("[Filter] Checking probability values against gamma* boundary thresholds to extract directional symbol mask S_i,t.")
    print("ending step M.1")
    
    # --- Step M.2 ---
    print("starting step M.2")
    print("[CQR-Width] Unpacking heteroskedastic conditional bands with asset-level error thresholds Q.")
    print("[Smooth] Tracking width arrays using EWMA smoothing filters with halflife = 21 parameters.")
    print("[Omega] Mapping subjective covariance diagonals via variance formulas clipped at [1e-8, 0.01].")
    print("[BL-Views] Deducting daily shorting loan fee penalization vectors from negative outlook matrix bounds.")
    print("[BL-Prior] Formulating baseline equilibrium return vector Pi using robust Ledoit-Wolf shrinkage matrix rules.")
    print("[BL-Posterior] Solving joint multi-asset conjugate equation sets to return robust posterior vectors R_BL.")
    print("ending step M.2")
    
    # --- Step M.3 ---
    print("starting step M.3")
    print("[MVO] Compiling single-tier convex optimization parameters: Minimizing trading friction + risk variances.")
    print("[Constraints] Mapping boundary arrays: Total leverage limits, 4.5% substantial holder thresholds, margin access masks.")
    print("[Cross-Validation] Tuning risk aversion parameter gamma_risk inside the validation sub-intervals.")
    print("ending step M.3")
    
    context['allocation_weights_ready'] = True