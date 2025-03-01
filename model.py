import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch

MODEL_PATH = '/model-repository/jina-embeddings-v2-base-en/1'

class TritonPythonModel:
    def initialize(self, args):
        self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).half()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Model initialized successfully.")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            input_array = input_tensor.as_numpy()  # Shape: [batch_size, 1]
            input_texts = [row[0].decode('utf-8') if isinstance(row[0], bytes) else row[0] 
                          for row in input_array]

            with torch.no_grad():
                inputs = self.tokenizer(input_texts, padding=True, truncation=True, 
                                      return_tensors="pt", max_length=256)  # Reduced max_length
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Shape: [batch_size, 768]
                torch.cuda.empty_cache()  # Free memory

            embeddings_array = embeddings.astype(np.float32)
            output_tensor = pb_utils.Tensor("embeddings", embeddings_array)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        print("Cleaning up Jina embeddings model...")
