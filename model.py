import triton_python_backend_utils as pb_utils
from transformers import AutoModel, AutoTokenizer  # Add tokenizer for batch processing
import numpy as np
import torch

MODEL_PATH = '/model-repository/jina-embeddings-v2-base-en/1'

class TritonPythonModel:
    def initialize(self, args):
        # Load the Jina model and tokenizer
        self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Model initialized successfully.")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get input tensor (expected shape: [batch_size, 1] for batched strings)
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            input_array = input_tensor.as_numpy()  # Shape: [batch_size, 1]

            # Convert batched NumPy array to list of strings
            # Handle [batch_size, 1] shape correctly
            input_texts = [row[0].decode('utf-8') if isinstance(row[0], bytes) else row[0] 
                          for row in input_array]

            # Tokenize and process the batch efficiently
            with torch.no_grad():
                # Tokenize all texts at once
                inputs = self.tokenizer(input_texts, padding=True, truncation=True, 
                                      return_tensors="pt", max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Generate embeddings in one forward pass
                outputs = self.model(**inputs)
                # Mean pooling to get fixed-size embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Shape: [batch_size, 768]

            # Create output tensor
            embeddings_array = embeddings.astype(np.float32)
            output_tensor = pb_utils.Tensor("embeddings", embeddings_array)

            # Create response
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        print("Cleaning up Jina embeddings model...")
