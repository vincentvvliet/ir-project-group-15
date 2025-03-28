from fast_forward.encoder.base import Encoder
from sentence_transformers import SentenceTransformer
import numpy as np

# Encoder class definition
class MiniLMEncoder(Encoder):
    def __init__(self, model_name="sentence-transformers/msmarco-MiniLM-L6-cos-v5"):
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: "Sequence[str]") -> "np.ndarray":
        """Encodes texts into embeddings using MiniLM."""
        return np.array(self.model.encode(texts, convert_to_numpy=True))