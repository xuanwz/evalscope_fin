from evalscope.run import run_task
from evalscope.config import TaskConfig




task_cfg = TaskConfig(
    model='glm-4-flash',
    eval_type='service',
    datasets=['gsm8k'],
    few_shot_num=0,
    dataset-args = {"gsm8k": {"few_shot_num": 0, "few_shot_random": false}},
    api_url='http://localhost:3000/v1/chat/completions',
    api_key='sk-aPDCGfJxbXE5rNzL08C0Fb4c62Ad41E3AcD91b42E637646e',
    limit=5
)

run_task(task_cfg=task_cfg)


"""
evalscope eval `
 --model glm-4-flash `
 --api-url http://localhost:3000/v1/chat/completions `
 --api-key sk-aPDCGfJxbXE5rNzL08C0Fb4c62Ad41E3AcD91b42E637646e `
 --eval-type service `
 --datasets gsm8k `
 --limit 5
"""