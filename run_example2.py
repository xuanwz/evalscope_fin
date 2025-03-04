from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

#先运行这个命令起一个vllm推理服务，可以根据需要修改参数
#CUDA_VISIBLE_DEVICES=0,1 export VLLM_USE_MODELSCOPE=True && python -m vllm.entrypoints.openai.api_server --model /root/FinR1/models/Qwen/Qwen2.5-3B-Instruct --served-model-name qwen2.5 --trust_remote_code --port 8801 --tensor-parallel-size 2

task_cfg = TaskConfig(
    model='qwen2.5',   # 模型名称 (需要与部署时的模型名称一致)
    api_url='http://127.0.0.1:8801/v1/chat/completions',  # 推理服务地址，8801是与上面对应的
    eval_type='service',   # 评测类型，SERVICE表示评测推理服务
    datasets=[
    # 'math_500',  # 数据集名称
    'fineval', 'fineval_definition'   
    ],
    dataset_args={ 
    'fineval_definition': {'subset_list': ['main'], 'few_shot_num': 0},
    },    
    limit=5,
    eval_batch_size=1,
    generation_config={       # 模型推理配置
        'max_tokens': 4096,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,   # 采样温度 (deepseek 报告推荐值)
        'top_p': 0.95,        # top-p采样 (deepseek 报告推荐值)
        'n': 1                # 每个请求产生的回复数量 (注意 lmdeploy 目前只支持 n=1)
    },
    stream=False               # 是否使用流式请求，推荐设置为True防止请求超时
)

run_task(task_cfg=task_cfg)