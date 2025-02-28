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
            optional_params["_original_input_list"] = input
            optional_params["_client_mode"] = True
            if len(input) > 0:
                return {"text": input[0]}
            else:
                return {"text": ""}
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

        first_embedding = raw_response_json["embeddings"]
        original_input_list = optional_params.get("_original_input_list", None)
        client_mode = optional_params.get("_client_mode", False)
        
        if not original_input_list or len(original_input_list) <= 1 or not client_mode:
            return first_embedding
        
        import copy
        all_embeddings = [first_embedding]
        
        for i in range(1, len(original_input_list)):
            input_text = original_input_list[i]
            
            try:
                new_request_data = copy.deepcopy(request_data)
                new_request_data["text"] = input_text
                
                new_response = httpx.post(
                    url=raw_response.request.url,
                    json=new_request_data,
                    headers=raw_response.request.headers,
                    timeout=30
                )
                
                if new_response.status_code == 200:
                    new_response_json = new_response.json()
                    new_embedding = new_response_json["embeddings"]
                    all_embeddings.append(new_embedding)
                else:
                    logging_obj.post_call(
                        input=input_text,
                        api_key=api_key,
                        original_response=str(new_response.text),
                        additional_args={"error": f"Failed to get embedding for input {i+1}"}
                    )
            except Exception as e:
                logging_obj.post_call(
                    input=input_text,
                    api_key=api_key,
                    original_response=str(e),
                    additional_args={"error": f"Exception processing input {i+1}"}
                )
        
        return all_embeddings

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
