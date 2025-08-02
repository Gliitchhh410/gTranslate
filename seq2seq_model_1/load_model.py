# load_model.py
import torch
import spacy
from model import Encoder, Decoder, Seq2Seq

# Special tokens (same as training)
BOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

def load_model_and_vocabs(device='cpu'):
    """Load the trained model and vocabularies"""
    
    print("Loading model components...")
    
    # Load model configuration
    model_config = torch.load('model_config.pt', map_location=device)
    print("✓ Model config loaded")
    
    # Load vocabularies
    german_vocab = torch.load('german_vocab.pt', map_location=device)
    english_vocab = torch.load('english_vocab.pt', map_location=device)
    print("✓ Vocabularies loaded")
    print(f"  German vocab size: {len(german_vocab)}")
    print(f"  English vocab size: {len(english_vocab)}")
    
    # Recreate model architecture
    encoder = Encoder(
        input_size=model_config['german_vocab_size'],
        embedding_size=model_config['encoder_embedding_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout_value=model_config['encoder_dropout']
    ).to(device)
    
    decoder = Decoder(
        input_size=model_config['english_vocab_size'],
        embedding_size=model_config['decoder_embedding_size'],
        hidden_size=model_config['hidden_size'],
        output_size=model_config['english_vocab_size'],
        num_layers=model_config['num_layers'],
        dropout_value=model_config['decoder_dropout']
    ).to(device)
    
    model = Seq2Seq(encoder, decoder).to(device)
    print("✓ Model architecture created")
    
    # Load trained weights
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.eval()
    print("✓ Trained weights loaded")
    
    return model, german_vocab, english_vocab

def load_spacy_models():
    """Load spaCy tokenizers"""
    print("Loading spaCy models...")
    spacy_ger = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    print("✓ spaCy models loaded")
    return spacy_ger, spacy_en

def tokenizer_de(text, spacy_ger):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_en(text, spacy_en):
    return [tok.text for tok in spacy_en.tokenizer(text)]