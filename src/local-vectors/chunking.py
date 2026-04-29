from typing import List
from tokenizers import Tokenizer


class SlidingWindowChunker:
    def __init__(self, tokenizer: Tokenizer, size: int = 1024, overlap: int = 128):
        self.tokenizer = tokenizer
        self.size = size
        self.overlap = overlap
        self.step = size - overlap


    def chunk_text(self, text: str) -> List[str]:
        # Encode string to token IDs.
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids
        
        chunks = []
        for i in range(0, len(tokens), self.step):
            chunk_ids = tokens[i : i + self.size]

            # Decode IDs back to text for embedding model.
            chunks.append(self.tokenizer.decode(chunk_ids))
            
            if i + self.size >= len(tokens):
                break
            
        return chunks