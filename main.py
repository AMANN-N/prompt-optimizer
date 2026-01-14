from src.orchestrator import Orchestrator
import json
import os
import yaml

def load_inputs(config_path="inputs.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # 0. Load Config
    try:
        config = load_inputs()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # 1. Setup LLM
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY not found in environment. Please set it to run with real Gemini calls.")
        # return

    llm_settings = config.get("llm", {})
    provider = llm_settings.get("provider", "gemini")
    model_name = llm_settings.get("model", "gemini-2.0-flash-lite-preview-02-05")
    
    orchestrator = Orchestrator(llm_provider=provider, api_key=api_key, model_name=model_name)
    
    # 2. User Intent
    intent = config.get("task_intent")
    if not intent:
        print("❌ Error: 'task_intent' missing in inputs.yaml")
        return

    # 3. Demo Data
    dataset_config = config.get("dataset", {})
    base_path = dataset_config.get("base_path", ".")
    examples = dataset_config.get("examples", [])
    
    demo_data = []
    for ex in examples:
        fname = ex.get("filename")
        if not fname: continue
            
        full_path = os.path.join(base_path, fname)
        if os.path.exists(full_path):
            demo_data.append({
                "input": full_path,
                "output": ex.get("ground_truth", {})
            })
        else:
            print(f"⚠️ Warning: Image not found: {full_path}")

    if not demo_data:
        print("❌ Error: No valid images found from inputs.yaml")
        return

    # 4. Run Optimization
    optimization_settings = config.get("optimization", {})
    max_iters = optimization_settings.get("max_iterations", 3)
    
    print(f"🚀 Starting optimization for: '{intent[:50]}...'")
    print(f"📂 Dataset: {len(demo_data)} examples")
    
    base_prompt = config.get("base_prompt")
    result = orchestrator.optimize_prompt(intent, demo_data, max_iterations=max_iters, base_prompt_context=base_prompt)
    
    print("\n\nFINAL RESULT:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
