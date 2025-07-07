"""
Simple runner script for Vacuum Review IReRa Classification with Gemini
"""

import dspy
from vacuum_irera import VacuumIReRa, VacuumIReRaConfig, load_vacuum_data, evaluate_program

def setup_dspy():
    """Configure DSPy with Gemini via liteLLM"""
    import os
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not found in environment")
        print("Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("Get API key from: https://aistudio.google.com/app/apikey")
        return False
    
    # Configure with Gemini via liteLLM
    lm = dspy.LM('gemini/gemini-2.0-flash')
    dspy.configure(lm=lm)
    return True

def run_pipeline():
    """Run the complete vacuum review classification pipeline"""
    
    if not setup_dspy():
        return
    
    print("🔧 Setting up Vacuum Review IReRa Pipeline with Gemini...")
    
    # Configuration  
    config = VacuumIReRaConfig()
    
    # Load data
    train_examples = load_vacuum_data(config.dataset_dev_path)
    test_examples = load_vacuum_data(config.dataset_test_path)
    
    print(f"📊 Loaded {len(train_examples)} training, {len(test_examples)} test examples")
    
    # Create and test program
    program = VacuumIReRa(config)
    
    # Quick test on first example
    sample = train_examples[0]
    prediction = program(sample.text)
    
    print("\n🧪 Sample Prediction:")
    print(f"Review: {sample.text[:100]}...")
    print(f"Predicted: {prediction.predictions[:3]}")
    print(f"Actual: {sample.label}")
    
    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    metrics = evaluate_program(program, test_examples)
    
    print("\nResults:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    return program, metrics

if __name__ == "__main__":
    run_pipeline()
