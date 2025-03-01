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
        # Send the input directly to Triton - our model can handle both single and batched inputs
        return {"text": input}

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

        # Get embeddings from response
        embeddings = raw_response_json["embeddings"]
        
        # Determine if we have multiple embeddings or a single embedding
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
            # Multiple embeddings - one for each input
            embedding_data = []
            for i, embedding in enumerate(embeddings):
                embedding_data.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding
                })
            
            # Set up model_response properly
            model_response.data = embedding_data
            model_response.model = model
            return model_response
            
        elif isinstance(embeddings, list) and len(embeddings) > 0:
            # Single embedding
            model_response.data = [{
                "object": "embedding",
                "index": 0,
                "embedding": embeddings
            }]
            model_response.model = model
            return model_response
            
        else:
            # Unexpected format
            return embeddings

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
