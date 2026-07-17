# Quant-ultra

## **On hold** to add logic:
 - Extra logging during phase sequence for more explanability
 - Remove redundent debugging purpose logger
 - Adjust data retriving part, seems using pre-settled & hard-coded now (from debugging log)
 - (Not sure) Model training maybe not updated because of cache loading mechanism(?)

## Work log by 2026/07/06:
- Function Update Log
  - Phase logger added
  - Debugging logger cleaned
  - Data retriving part with extra resourse
 - Schedule Work Log
   - Simplify logger -> dual language into 1 line logger, even less debugging purpose loggers
   - Some logger error message from compilor need attention, check phase result
   - Check data download truth (ensuring download success + checking data is true and match by date)
- Future Idea & Long Term Planning log
  - Convert into C++ version for higher execution speed
  - Apply multi-threading and multi-laptop co-working mechanism
  - New Phase 10 for generating a visualized report for current execution round
  - New Phase 11 for applying local LLM to read and explain report from Phase 10

## Work schedlue by 2026/07/10:
- Function Update Log
  - Logger cleaned with less message
  - **CPU / GPU acclerate for training added**
  - For quick tesing purpose, training period is last 3 days only
  - Current speed for entire pipeline (3 day training) is about 1.5h without terminate
    - Old cache of Phase Result was cleaned before running the main engine
 - Schedule Work Log
   - Find a way to get US market data with stable source and real data
    - yahoo seems VPN settle required
   - Adjust logger file into new structure
     -   -> Better to be like (Folder: log) - (Folder: #Date - # Time-Stamp) - (Folder: #Phase number)
   - Try to convert into C++ version but keeping python flexibilty (seems has a library, research required)
- Future Idea & Long Term Planning log
  - New Phase 10 for generating a visualized report for current execution round
  - New Phase 11 for applying local LLM to read and explain report from Phase 10
  - Apply multi-threading and multi-laptop co-working mechanism
    - **CPU/GPU accleration is added, lower priority**

## Work schedlue by 2026/07/17:
- Function Update Log
  - Fuuly pass for Phase_0(Main) to Phase_9
  - Phase_10 LLM draft is placed
  - Enhance for Phase 1~9 is placed in dlc folder
  - Loop re-verify for doing 2nd pass is placed in dlc folder
  - Auto adjust metrics script is placed in dlc folder
 - Schedule Work Log
   - Debug for Phase 10
   - Debug for auto metic
   - Debug for dlc enhance
   - Debug for 2nd pass
- Future Idea & Long Term Planning log
  - Cython re-structure after all works above finish
