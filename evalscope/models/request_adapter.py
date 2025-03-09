import requests
from evalscope.models.base_adapter import BaseModelAdapter
from typing import List, Union, Optional
import json
import time

from evalscope.utils.logger import get_logger

logger = get_logger()

class RequestModelAdapter(BaseModelAdapter):
    """
    Request model adapter to request remote API model and generate results using requests.
    """

    def __init__(self, api_url: str, model_id: str, api_key: str = 'EMPTY', **kwargs):
        self.api_url = api_url
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

    def send_request_with_retry(self, request_json: dict, retry_count: int = 3) -> Optional[dict]:
        """使用 requests 库发送请求并处理响应，带重试逻辑。"""
        headers = self.headers if self.headers else {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # 设置严格的超时限制
        timeout = (100, 300)  # (连接超时, 读取超时)

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=request_json,
                    stream=self.stream,
                    timeout=timeout
                )
                response.raise_for_status()
                
                if self.stream:
                    return self._collect_stream_response(response)
                return response.json()
                
            except Exception as e:
                logger.error(f'请求错误 (尝试 {attempt + 1}/{retry_count}): {str(e)}')
                if attempt < retry_count - 1:
                    wait_time = min(2 ** attempt, 30)  # 最大等待30秒
                    time.sleep(wait_time)
                else:
                    # 最后一次重试失败，返回 None 而不是抛出异常
                    logger.error(f'所有重试均失败，跳过该请求')
                    return None

    def _collect_stream_response(self, response) -> dict:
        """Collect and process stream response."""
        try:
            # 尝试直接解析整个响应
            if hasattr(response, 'json'):
                return response.json()
        except:
            pass
            
        # 如果不是完整JSON，则按流式处理
        collected_content = ""
        reasoning_content = ""
        has_reasoning = False
        usage = None
        
        for line in response.iter_lines():
            if line:
                # 移除 "data: " 前缀并解析 JSON
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    json_str = line_text[6:]  # 跳过 "data: " 前缀
                    if json_str.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(json_str)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                collected_content += delta['content']
                            if 'reasoning_content' in delta:
                                has_reasoning = True
                                reasoning_content += delta['reasoning_content']
                        
                        # 收集使用情况统计（如果有）
                        if 'usage' in chunk:
                            usage = chunk['usage']
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON解析错误: {e}, 行内容: {json_str}")
                else:
                    # 尝试直接解析整行为JSON
                    try:
                        return json.loads(line_text)
                    except:
                        logger.debug(f"无法解析行: {line_text}")
        
        # 构建与非流式响应格式一致的结构
        result = {
            'choices': [{
                'message': {
                    'content': collected_content
                }
            }]
        }
        
        # 如果存在reasoning_content，则添加到结果中
        if has_reasoning:
            result['choices'][0]['message']['reasoning_content'] = reasoning_content
        
        if usage:
            result['usage'] = usage
            
        return result

    def handle_request_error(self, e: Exception):
        logger.error(f'Error when calling API: {str(e)}')
        raise 