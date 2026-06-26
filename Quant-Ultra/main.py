"""
Quant-Ultra + Conformal-BL Investment Workflow Engine
Main Orchestrator [2026 Production Release]
"""

import sys

# Import pipeline steps
import step1_data_foundation
import step2_data_slicing
import step3_pit_setup
import step4_labeling_weighting
import step5_model_training_calibration
import step6_position_sizing
import step7_fsm_backtest
import step8_audit_stress_test
import step9_live_mlops

def run_pipeline():
    print("====================================================")
    print("LAUNCHING QUANT-ULTRA + CONFORMAL-BL PRODUCTION FLOW")
    print("====================================================\n")
    
    # Context dictionary to pass state variables down the stream
    pipeline_context = {}
    
    # Phase 1: Data Foundation
    step1_data_foundation.execute(pipeline_context)
    
    # Phase 2: Data Slicing & Isolation
    step2_data_slicing.execute(pipeline_context)
    
    # Phase 3: Point-in-Time Setup
    step3_pit_setup.execute(pipeline_context)
    
    # Phase 4: Labeling & Weighting
    step4_labeling_weighting.execute(pipeline_context)
    
    # Phase 5: CV, Dual-Calib & Model Fitting
    step5_model_training_calibration.execute(pipeline_context)
    
    # Phase 6: Multi-Asset Position Sizing & Convex Optimization
    step6_position_sizing.execute(pipeline_context)
    
    # Phase 7: State-Machine Backtesting
    step7_fsm_backtest.execute(pipeline_context)
    
    # Phase 8: DSR Final Audit & Macro Stress Test
    step8_audit_stress_test.execute(pipeline_context)
    
    # Phase 9: Live Deploy & MLOps Operations
    step9_live_mlops.execute(pipeline_context)
    
    print("\n====================================================")
    print("QUANT-ULTRA WORKFLOW COMPLETED SUCCESSFULLY")
    print("====================================================")

if __name__ == '__main__':
    run_pipeline()