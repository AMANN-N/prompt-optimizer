# Prompt Optimization Agent

## Overview
This system is a **Prompt Compiler + Test Harness + Optimizer**. It converts unstructured user intent into a clean, optimized, and frozen prompt for information extraction tasks.

## Structure
- `src/agents`: Contains the specialized agents for each step of the pipeline.
- `src/core`: Shared definitions and utilities.
- `src/orchestrator.py`: Manages the flow of the optimization loop.

## How It Works
1. **Configuration**: You define your goal (Intent) and provide a few examples (Input Image + Ground Truth) in `inputs.yaml`.
2. **Orchestration**: Run `python main.py`. The system:
   - **Formalizes** your intent into a structured task definition.
   - **Generates** an initial "Base Prompt".
   - **Iterates** (Optimization Loop):
     - **Executes** the prompt on your examples using Gemini 2.0.
     - **Evaluates** the results against your Ground Truth.
     - **Analyzes** failures (hallucinations, missing fields, wrong format).
     - **Optimizes** the prompt to fix those specific failures.
   - **Freezes** the best-performing prompt.

## Dataset & Ground Truth Guide
**Q: Do I need ground truth for all 100 images?**
**A: No!** 

For **Optimization**, you only need a **small, broadly representative subset** (e.g., 5-10 images).
- **Purpose**: These act as a "Training Set" for the agent to learn the best prompt structure.
- **Strategy**: Pick the most diverse or "difficult" layouts from your 100 images.
- **Workflow**: 
  1. Add 5-10 hard examples to `inputs.yaml`.
  2. Run the optimizer to get a high-quality frozen prompt.
  3. Use that **Frozen Prompt** on the remaining 90+ images (Production Mode) without needing ground truth.
