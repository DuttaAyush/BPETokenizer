from tokenizer import BPETokenizer

if __name__ == "__main__":
    tokenizer = BPETokenizer(num_merges=100)
    
    with open("../Alice.txt", "r", encoding="utf-8") as f:
        corpus = f.read()
    
    tokenizer.train(corpus)
    
    print(f"\nLearned merges: {tokenizer.merges}")
    
    print(f"\nVocabulary size: {len(tokenizer.vocab)}")
    print("Sample vocabulary entries:")
    for i, (id, token) in enumerate(list(tokenizer.vocab.items())[:10]):
        print(f"  {id}: '{token}'")
    
    with open("../output.txt","w", encoding="utf-8") as f:
        test_text = "Alice was beginning to get very tired."
        print(f"\nOriginal text: '{test_text}'")
        f.write(f"\nOriginal text: '{test_text}' \n")
        
        encoded = tokenizer.encode(test_text)
        print(f"Encoded: {encoded}")
        f.write(f"Encoded: {encoded} \n")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: '{decoded}'")
        f.write(f"Decoded: '{decoded}' \n")


    