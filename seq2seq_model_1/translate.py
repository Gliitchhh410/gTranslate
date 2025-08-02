# translate.py
import torch
from load_model import BOS_TOKEN, EOS_TOKEN, tokenizer_de

def translate_sentence(sentence, model, german_vocab, english_vocab, spacy_ger, device, max_length=50):
    """
    Translate a German sentence to English
    """
    model.eval()

    # Tokenize (keep original case - this was the fix!)
    tokens = tokenizer_de(sentence, spacy_ger)
    print(f"Tokens: {tokens}")
    
    # Convert to indices
    numericalized = []
    unk_count = 0
    for token in tokens:
        try:
            idx = german_vocab[token]
            numericalized.append(idx)
        except KeyError:
            numericalized.append(german_vocab[UNK_TOKEN])
            unk_count += 1
    
    print(f"Unknown tokens: {unk_count}/{len(tokens)}")
    
    # If too many unknowns, warn user
    if unk_count > len(tokens) // 2:
        print("⚠️  Warning: Many unknown words, translation quality may be poor")

    # Convert to tensor and add batch dimension
    sentence_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)

    # Encode the input
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    # Start decoding with <sos> token
    outputs = [english_vocab[BOS_TOKEN]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Stop if predicted <eos>
        if best_guess == english_vocab[EOS_TOKEN]:
            break

    # Convert indices to tokens
    translated_tokens = [english_vocab.get_itos()[idx] for idx in outputs]

    # Remove <sos> and <eos> for display
    return translated_tokens[1:-1]