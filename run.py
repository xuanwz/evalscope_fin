from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

task_cfg = TaskConfig(
    model='glm-4-flash',   # 模型名称 (需要与部署时的模型名称一致)
    api_url='http://localhost:3000/v1/chat/completions',  # 推理服务地址
    api_key='sk-aPDCGfJxbXE5rNzL08C0Fb4c62Ad41E3AcD91b42E637646e',
    eval_type='service',   # 评测类型，SERVICE表示评测推理服务
    datasets=[
    # 'math_500',  # 数据集名称
    'fineval_definition'   
    ],
    dataset_args={ # EvalScope内置支持，无需指定数据集ID
    'fineval_definition': {'subset_list': ['main'], 'few_shot_num': 0},
    },    
    limit=5,
    eval_batch_size=5,
    generation_config={       # 模型推理配置
        'max_tokens': 4096,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 1                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
    stream=False               # 是否使用流式请求，推荐设置为True防止请求超时
)

run_task(task_cfg=task_cfg)