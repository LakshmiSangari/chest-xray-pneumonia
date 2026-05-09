from src.train import train
from src.evaluate import evaluate
 
if __name__ == "__main__":
    print("🚀 Starting Training...")
    train()
 
    print("\n📊 Starting Evaluation...")
    evaluate()
 
    print("\n✅ All Done! Check results/ folder for plots.")