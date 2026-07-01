# Quant-ultra

## Folder - Phase-pipeline-flow 

 - Stores the original and start up version
 - Iterated from project "ML-Quant-A-stock"
 - Old version code projects and the general project idea
 - For saving purpose only, not for use

## Folder - Split-up

- Stores the splited version of code
- All files are seperated and splited from the old "Phase-pipeline-flow" code
- Code files are more moduleized, easier for locating bug and maintaince
- **On hold** to add logic for each session/phase:
    - 1) During each phase initialization:
        - Check finished result cache 
            - if **not exist**      -> process normally, store result in new folder by date
            - if **Update Required**-> process normally, store result in new folder by date
            - if **Update Not Need**-> load stored file, skip current process step/sub-step.
    - 2) Saving phase/sessional result, saving general processing time
        - After normal process progress, save result to local folders
        - Saving format: 
            - parquet (for python script)
            - feather (for future C language)
        - Saving path:
            - Split-Up (Root) 
                - Phase Result
                    - parquet
                        - Phase 1
                            - Step 1.1
                            - Step 1.2
                            - ...
                        - Phase 2
                        - ...
                    - feather
                        - Phase 1
                        - Phase 2
                        - ...
                - Main
                - Step 1
                - Step 2
                - Step 3
                - ...