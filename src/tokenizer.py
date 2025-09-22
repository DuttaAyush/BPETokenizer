# Class for tokenizer made
class BPETokenizer:
    # init members
    def __init__(self, num_merges=50):
        self.num_merges = num_merges #no of times the words in vocab to be merged higher no means higher accuracy etc
        self.vocab = {}
        self.inv_vocab = {}
        self.merges = []

    def get_stats(self, tokens):
        """Count frequency of adjacent symbol pairs"""
        # Give the count of freq.
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
        bigram = ' '.join(pair)  
        replacement = ''.join(pair) #Merged Tokens
        
        for word, freq in tokens.items():
            new_word = word.replace(bigram, replacement)
            new_tokens[new_word] = freq
        return new_tokens

    def train(self, corpus):
        """Train the BPE tokenizer on a corpus"""
        tokens = {}
        for word in corpus.split():
            #Split the words
            word_tokens = ' '.join(list(word)) + ' </w>'
            tokens[word_tokens] = tokens.get(word_tokens, 0) + 1

        # perform merging
        for i in range(self.num_merges):
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            #Finding most freq pair
            best = max(pairs, key=pairs.get)
            #Merge pairs
            tokens = self.merge_vocab(best, tokens)
            self.merges.append(best)
        #final Vocab building
        all_symbols = set()
        for word_tokens in tokens.keys():
            all_symbols.update(word_tokens.split())
        #Save the vocab with mappings
        self.vocab = {i: sym for i, sym in enumerate(sorted(all_symbols))}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        """Encode text into token IDs"""
        if not text.strip(): #Handle empty input
            return []
        #Split them
        words = text.split()
        all_token_ids = []
        
        for word in words:
            tokens = list(word) + ['</w>']
            # apply from new learned words
            for merge in self.merges:
                i = 0
                while i < len(tokens) - 1:
                    if (i < len(tokens) - 1 and 
                        tokens[i] == merge[0] and 
                        tokens[i + 1] == merge[1]):
                        tokens[i:i+2] = [''.join(merge)]
                    else:
                        i += 1
            # Save the tokens as ids
            word_ids = []
            for token in tokens:
                if token in self.inv_vocab:
                    word_ids.append(self.inv_vocab[token])
            
            all_token_ids.extend(word_ids)
        
        return all_token_ids

    def decode(self, ids):
        """Decode token IDs back to text"""
        if not ids:
            return ""
        # Get tokens from id
        tokens = []
        for token_id in ids:
            if token_id in self.vocab:
                tokens.append(self.vocab[token_id])
        
        # Join the tokens
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()


