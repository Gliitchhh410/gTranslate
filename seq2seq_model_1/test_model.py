# test_model.py
import torch
from load_model import load_model_and_vocabs, load_spacy_models
from translate import translate_sentence

def test_model():
    """Test that everything loads and works correctly"""
    
    print("🧪 Testing local model setup...")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load everything
        model, german_vocab, english_vocab = load_model_and_vocabs(device)
        spacy_ger, spacy_en = load_spacy_models()
        
        print("\n✅ All components loaded successfully!")
        
        # Test translations
        test_sentences = [
            "Ein Mann läuft auf der Straße.",
            "Eine Frau geht die Straße entlang.",
            "Kinder laufen auf dem Spielplatz."
        ]
        
        print("\n🔄 Testing translations:")
        print("-" * 30)
        
        for sentence in test_sentences:
            translation = translate_sentence(
                sentence, model, german_vocab, english_vocab, spacy_ger, device
            )
            print(f"DE: {sentence}")
            print(f"EN: {' '.join(translation)}")
            print()
        
        print("✅ Local setup working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_model()