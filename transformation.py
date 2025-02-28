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
        if isinstance(input, str):
            return {"text": input}
        elif isinstance(input, list):
            # For list input, we only process one item at a time
            # Store original list for sequential processing
            optional_params["_original_input_list"] = input
            optional_params["_is_batched_request"] = True
            optional_params["_current_input_index"] = 0
            
            # Only process the first item in this request
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

        # Extract the raw embedding
        current_embedding = raw_response_json["embeddings"]
        
        # Check if this is a batched request
        is_batched = optional_params.get("_is_batched_request", False)
        
        if not is_batched:
            # For single input, return the raw embedding directly
            return current_embedding
        
        # For batched requests, process remaining inputs if necessary
        original_input_list = optional_params.get("_original_input_list", [])
        current_index = optional_params.get("_current_input_index", 0)
        
        # If we're processing the last item, return just this embedding
        if current_index >= len(original_input_list) - 1:
            return current_embedding
        
        # Otherwise, process the next item and update the index
        next_index = current_index + 1
        if next_index < len(original_input_list):
            next_input = original_input_list[next_index]
            
            try:
                # Make a new request for the next input
                next_request_data = {"text": next_input}
                
                # Make the request to the same endpoint
                next_response = httpx.post(
                    url=raw_response.request.url, 
                    json=next_request_data,
                    headers=raw_response.request.headers,
                    timeout=30
                )
                
                if next_response.status_code == 200:
                    # Get the next embedding
                    next_response_json = next_response.json()
                    next_embedding = next_response_json["embeddings"]
                    
                    # Update optional params for next iteration
                    optional_params["_current_input_index"] = next_index
                    
                    # Call ourselves recursively to process the next input
                    return self.transform_embedding_response(
                        model=model,
                        raw_response=next_response,
                        model_response=model_response,
                        logging_obj=logging_obj,
                        api_key=api_key,
                        request_data=next_request_data,
                        optional_params=optional_params,
                        litellm_params=litellm_params
                    )
                else:
                    # If there's an error, just return the current embedding
                    logging_obj.post_call(
                        input=next_input, 
                        api_key=api_key, 
                        original_response=str(next_response.text),
                        additional_args={"error": f"Failed to process next input"}
                    )
                    return current_embedding
            except Exception as e:
                # Log the error and return the current embedding
                logging_obj.post_call(
                    input=next_input, 
                    api_key=api_key, 
                    original_response=str(e),
                    additional_args={"error": "Exception processing next input"}
                )
                return current_embedding
        
        # Fallback: return the current embedding
        return current_embedding

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
