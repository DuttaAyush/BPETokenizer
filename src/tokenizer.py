class BPETokenizer:
    def __init__(self, num_merges=50):
        self.num_merges = num_merges
        self.vocab = {}
        self.inv_vocab = {}
        self.merges = []

    def get_stats(self, tokens):
        """Count frequency of adjacent symbol pairs"""
        pairs = {}
        for word, freq in tokens.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def merge_vocab(self, pair, tokens):
        """Merge the most frequent pair in vocabulary"""
        new_tokens = {}
        bigram = ' '.join(pair)  # Space-separated pair
        replacement = ''.join(pair)  # Merged token
        
        for word, freq in tokens.items():
            new_word = word.replace(bigram, replacement)
            new_tokens[new_word] = freq
        return new_tokens

    def train(self, corpus):
        """Train the BPE tokenizer on a corpus"""
        # Initialize token dictionary with character-split words
        tokens = {}
        for word in corpus.split():
            # Split word into characters and add end-of-word token
            word_tokens = ' '.join(list(word)) + ' </w>'
            tokens[word_tokens] = tokens.get(word_tokens, 0) + 1

        # Perform BPE merges
        for i in range(self.num_merges):
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            
            # Find most frequent pair
            best = max(pairs, key=pairs.get)
            
            # Merge the pair in vocabulary
            tokens = self.merge_vocab(best, tokens)
            self.merges.append(best)

        # Build final vocabulary from all unique symbols
        all_symbols = set()
        for word_tokens in tokens.keys():
            all_symbols.update(word_tokens.split())
        
        # Create vocabulary mappings
        self.vocab = {i: sym for i, sym in enumerate(sorted(all_symbols))}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        """Encode text into token IDs"""
        # Handle empty input
        if not text.strip():
            return []
        
        # Split text into words
        words = text.split()
        all_token_ids = []
        
        for word in words:
            # Start with character-level tokens
            tokens = list(word) + ['</w>']
            
            # Apply learned merges in order
            for merge in self.merges:
                i = 0
                while i < len(tokens) - 1:
                    if (i < len(tokens) - 1 and 
                        tokens[i] == merge[0] and 
                        tokens[i + 1] == merge[1]):
                        # Merge the pair
                        tokens[i:i+2] = [''.join(merge)]
                    else:
                        i += 1
            
            # Convert tokens to IDs
            word_ids = []
            for token in tokens:
                if token in self.inv_vocab:
                    word_ids.append(self.inv_vocab[token])
                # Skip unknown tokens (could also handle differently)
            
            all_token_ids.extend(word_ids)
        
        return all_token_ids

    def decode(self, ids):
        """Decode token IDs back to text"""
        if not ids:
            return ""
        
        # Convert IDs to tokens
        tokens = []
        for token_id in ids:
            if token_id in self.vocab:
                tokens.append(self.vocab[token_id])
        
        # Join tokens and handle end-of-word markers
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()


