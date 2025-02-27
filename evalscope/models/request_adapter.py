import requests
from evalscope.models.base_adapter import BaseModelAdapter
from typing import List, Union, Optional

from evalscope.utils.logger import get_logger

logger = get_logger()

class RequestModelAdapter(BaseModelAdapter):
    """
    Request model adapter to request remote API model and generate results using requests.
    """

    def __init__(self, api_url: str, model_id: str, api_key: str = 'EMPTY', **kwargs):
        self.api_url = api_url.rstrip('/').rsplit('/chat/completions', 1)[0]
        self.model_id = model_id
        self.api_key = api_key

        self.seed = kwargs.get('seed', None)
        self.headers = kwargs.get('headers', None)
        self.timeout = kwargs.get('timeout', 60)
        self.stream = kwargs.get('stream', False)
        self.model_cfg = {'api_url': api_url, 'model_id': model_id, 'api_key': api_key}

        super().__init__(model=None, model_cfg={'api_url': api_url, 'model_id': model_id, 'api_key': api_key}, **kwargs)

    def predict(self, inputs: List[Union[str, dict, list]], infer_cfg: dict = None) -> List[dict]:
        """
        Model prediction function using requests.
        """
        infer_cfg = infer_cfg or {}
        results = []

        for input_item in inputs:
            response = self.process_single_input(input_item, infer_cfg)
            results.append(response)

        return results

    def process_single_input(self, input_item: dict, infer_cfg: dict) -> dict:
        """Process a single input item."""
        data: list = input_item['data']
        query = data[0] if isinstance(data, list) else data
        system_prompt = input_item.get('system_prompt', None)

        content = self.make_request_content(query, system_prompt)
        request_json = self.make_request(content, infer_cfg)
        response = self.send_request_with_retry(request_json)  # 使用重试逻辑
        return response

    def make_request_content(self, query: str, system_prompt: Optional[str] = None) -> dict:
        """Make request content for the API."""
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': query})
        return messages

    def make_request(self, content: list, infer_cfg: dict = {}) -> dict:
        """Make request to remote API."""
        request_json = {
            'model': self.model_id,
            'messages': content,
            **infer_cfg
        }

        if 'timeout' in infer_cfg:
            request_json['timeout'] = infer_cfg['timeout']

        if 'stream' in infer_cfg:
            if self.stream:
                request_json['stream'] = self.stream
                request_json['stream_options'] = {'include_usage': True}

        return request_json

    def send_request_with_retry(self, request_json: dict, retry_count: int = 5) -> dict:
        """使用 requests 库发送请求并处理响应，带重试逻辑。"""
        headers = self.headers if self.headers else {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        for attempt in range(retry_count):
            try:
                response = requests.post(self.api_url + '/chat/completions', headers=headers, json=request_json, stream=self.stream)
                response.raise_for_status()  # 对于错误响应抛出异常
                
                if self.stream:
                    return self._collect_stream_response(response)
                return response.json()
            except requests.exceptions.HTTPError as http_err:
                logger.error(f'HTTP error occurred: {http_err}')  # 记录 HTTP 错误
                if attempt < retry_count - 1:
                    logger.info(f'正在重试... 第 {attempt + 2} 次尝试')
                else:
                    raise
            except Exception as e:
                logger.error(f'An error occurred: {e}')  # 记录其他错误
                if attempt < retry_count - 1:
                    logger.info(f'正在重试... 第 {attempt + 2} 次尝试')
                else:
                    raise

    def _collect_stream_response(self, response) -> dict:
        """Collect and process stream response."""
        collected_messages = []
        for line in response.iter_lines():
            if line:
                message = line.decode('utf-8')
                collected_messages.append(message)

        # 如果没有使用流式，返回的结构需要包含choices
        return {
            'choices': [{
                'message': {
                    'content': ''.join(collected_messages)
                }
            }]
        }  # 返回处理后的消息，包含choices结构
    def handle_request_error(self, e: Exception):
        logger.error(f'Error when calling API: {str(e)}')
        raise 