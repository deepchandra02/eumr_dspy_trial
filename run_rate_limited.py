"""
Rate-limited version for Gemini free tier
"""

import dspy
import time
from vacuum_irera import (
    VacuumIReRa,
    VacuumIReRaConfig,
    load_vacuum_data,
    evaluate_program,
)


def setup_dspy():
    """Configure DSPy with Gemini + rate limiting"""
    import os

    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not found")
        return False

    # Add rate limiting for free tier
    lm = dspy.LM("gemini/gemini-2.5-flash", max_tokens=150)
    dspy.configure(lm=lm)
    return True


def evaluate_with_rate_limit(program, examples, max_examples=5):
    """Evaluate with delays to respect rate limits"""
    results = []

    for i, example in enumerate(examples[:max_examples]):
        if i > 0:
            print(f"Waiting 6 seconds to respect rate limits...")
            time.sleep(6)  # 6 second delay between requests

        try:
            prediction = program(example.text)
            results.append(
                {
                    "predicted": prediction.predictions,
                    "actual": example.label,
                    "text": example.text[:100] + "...",
                }
            )
            print(f"Example {i+1}: ✓")
        except Exception as e:
            print(f"Example {i+1}: Failed - {str(e)[:100]}...")
            if "429" in str(e):
                print("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)

    return results


def run_limited_test():
    if not setup_dspy():
        return

    print("🔧 Running Limited Test (Rate-Limit Safe)...")

    config = VacuumIReRaConfig()
    train_examples = load_vacuum_data(config.dataset_dev_path)
    test_examples = load_vacuum_data(config.dataset_test_path)

    program = VacuumIReRa(config)

    # Test just 3 examples with delays
    print("\n📊 Testing 3 examples with rate limiting...")
    results = evaluate_with_rate_limit(program, test_examples, max_examples=3)

    print(f"\n🎯 Results:")
    for i, result in enumerate(results):
        print(f"\nExample {i+1}:")
        print(f"  Text: {result['text']}")
        print(f"  Predicted: {result['predicted'][:3]}")
        print(f"  Actual: {result['actual']}")

        # Simple accuracy check
        predicted_set = set(result["predicted"][:3])
        actual_set = set(result["actual"])
        overlap = len(predicted_set.intersection(actual_set))
        print(
            f"  Accuracy: {overlap}/{len(actual_set)} = {overlap/len(actual_set):.2f}"
        )


if __name__ == "__main__":
    run_limited_test()
