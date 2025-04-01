from fast_forward.encoder.base import Encoder
import numpy as np
from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-Vincent-L12-H384")
# model = AutoModel.from_pretrained("microsoft/Multilingual-Vincent-L12-H384")

# Encoder class definition
class MiniLMEncoder(Encoder):
    def __init__(self, model_name="microsoft/Multilingual-MiniLM-L12-H384"):
        self.model = AutoTokenizer.from_pretrained(model_name)

    def _encode(self, texts: "Sequence[str]") -> "np.ndarray":
        """Encodes texts into embeddings using Vincent."""
        return np.array(self.model(texts, convert_to_numpy=True))