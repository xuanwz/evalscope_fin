from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='deepseek-r1-distill-qwen-7b',   # 模型名称 (需要与部署时的模型名称一致)
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',  # 推理服务地址
    api_key='sk-40a2cf1184a74bcc9e74b53155e665a5',
    api_type='request',
    eval_type='service',   # 评测类型，SERVICE表示评测推理服务
    datasets=[
    # 'math_500',  # 数据集名称
    'senti_data','FinQA','ConvFinqa'
    ],  
    limit=1000, #每个数据集测试的数量
    eval_batch_size=3,
    generation_config={       # 模型推理配置
        'max_tokens': 4096,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 1                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
    stream=False               # 是否使用流式请求，推荐设置为True防止请求超时
)

run_task(task_cfg=task_cfg)