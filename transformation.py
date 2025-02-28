from typing import List, Optional, Union
import httpx
from litellm.llms.base_llm.chat.transformation import AllMessageValues, BaseLLMException
from litellm.llms.base_llm.embedding.transformation import (
    BaseEmbeddingConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllEmbeddingInputValues
from litellm.types.utils import EmbeddingResponse
from ..common_utils import TritonError

class TritonEmbeddingConfig(BaseEmbeddingConfig):
    """
    Transformations for triton /embeddings endpoint (This is a trtllm model)
    """
    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self, model: str) -> list:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to Triton Embedding params
        """
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return {}

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        # Check if input is a single string or a list of strings
        if isinstance(input, str):
            # Handle single string case
            return {"text": input}
        elif isinstance(input, list):
            # Store original input in optional_params for later use in response transform
            optional_params["_original_input_list"] = input
            
            # Process each input separately by handling only the first one here
            # We'll make multiple requests in transform_embedding_response
            return {"text": input[0] if len(input) > 0 else ""}
        else:
            raise ValueError(f"Unexpected input type: {type(input)}. Expected string or list of strings.")

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> EmbeddingResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise TritonError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        
        # Get the embeddings from the response
        _outputs = raw_response_json["embeddings"]
        
        # Check if this was part of a batch request (list of strings)
        original_input_list = optional_params.get("_original_input_list", None)
        
        # If we have a list of inputs but we're only processing the first one...
        if original_input_list and len(original_input_list) > 1:
            # Process just the first input now and handle the rest recursively
            first_embedding = _outputs
            
            # Create first embedding output
            _embedding_output = [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": first_embedding,
                }
            ]
            
            # Process the remaining inputs
            import httpx
            import copy
            remaining_inputs = original_input_list[1:]
            
            # For each remaining input, make a new request using the same client
            for i, input_text in enumerate(remaining_inputs):
                # Create a new request for each remaining input
                try:
                    # Create a copy of the original request with new input
                    new_request_data = copy.deepcopy(request_data)
                    new_request_data["text"] = input_text
                    
                    # Use the same client and headers as the original request
                    new_response = httpx.post(
                        url=raw_response.request.url, 
                        json=new_request_data,
                        headers=raw_response.request.headers,
                        timeout=30  # Adjust timeout as needed
                    )
                    
                    if new_response.status_code == 200:
                        new_embedding = new_response.json()["embeddings"]
                        # Add this embedding to our results
                        _embedding_output.append({
                            "object": "embedding",
                            "index": i + 1,  # Index starts from 1 for remaining inputs
                            "embedding": new_embedding,
                        })
                    else:
                        # Handle error for this specific input
                        logging_obj.post_call(
                            input=input_text, 
                            api_key=api_key, 
                            original_response=str(new_response.text),
                            additional_args={"error": f"Failed to process input at index {i+1}"}
                        )
                except Exception as e:
                    # Log error but continue processing remaining inputs
                    logging_obj.post_call(
                        input=input_text, 
                        api_key=api_key, 
                        original_response=str(e),
                        additional_args={"error": f"Exception processing input at index {i+1}"}
                    )
        else:
            # Normal processing for single input or when Triton handles batches correctly
            if isinstance(_outputs, list) and all(isinstance(emb, list) for emb in _outputs):
                # We have a batch of embeddings
                _embedding_output = [
                    {
                        "object": "embedding",
                        "index": idx,
                        "embedding": embedding,
                    }
                    for idx, embedding in enumerate(_outputs)
                ]
            else:
                # We have a single embedding result
                _embedding_output = [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": _outputs,
                    }
                ]
        
        # Set the model and data in the response
        model_response.model = raw_response_json.get("model_name", model)
        model_response.data = _embedding_output
        
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return TritonError(
            message=error_message, status_code=status_code, headers=headers
        )

    @staticmethod
    def split_embedding_by_shape(
        data: List[float], shape: List[int]
    ) -> List[List[float]]:
        if len(shape) != 2:
            raise ValueError("Shape must be of length 2.")
        embedding_size = shape[1]
        return [
            data[i * embedding_size : (i + 1) * embedding_size] for i in range(shape[0])
        ]
