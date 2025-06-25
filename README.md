# Experiment on Multimodal Infection

## Core Idea
1. Text trigger only: Only use text trigger, completely unrelated to any image operation
2. Training: Only train the text part of LLaVA, freeze the Vision Tower and MM Projector
3. Dataset: Directly use the datasets they provide, avoid generating ourselves
4. Training Process: Ensure following the LLaVA official training process


### Jun 24 TO-DO
1. Verification Experiment: Use clean images to check if only the corresponding three rows or three columns are modified
2. Test task: Jailbreak
3. Test task: Negsentiment
4. Test task: Refusal

### UPDATE:
- Input_ids desc: /research/inputids.md

**Doing**: Verification Task

