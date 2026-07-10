# Quant-ultra

## **On hold** to add logic:
 - Extra logging during phase sequence for more explanability
 - Remove redundent debugging purpose logger
 - Adjust data retriving part, seems using pre-settled & hard-coded now (from debugging log)
 - (Not sure) Model training maybe not updated because of cache loading mechanism(?)

### **Above logic has been fully implemented, below is new scheduled working list by 2026/7/6

## **On hold** to add logic:
 - Simplify logger -> dual language into 1 line logger, even less debugging purpose loggers
 - some logger error message from compilor need attention, check phase result
 - Check data download truth (ensuring download success + checking data is true and match by date)
 - New Phase 10 for generating a visualized report for current execution round
 - (below is long term designing)
  - Convert into C++ version for higher execution speed
  - Apply multi-threading and multi-laptop co-working mechanism
  - New Phase 11 for applying local LLM to read and explain report from Phase 10
