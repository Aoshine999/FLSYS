# simple_wandb_test.py
import wandb
import time
import os
import sys



print(">>> Initializing W&B")
try:
    # 确保 WANDB_MODE=offline 环境变量已设置，或在此处设置 mode='offline'
    run = wandb.init(project="test-offline-log", config={"test": 123},mode='offline'
                     ,settings=wandb.Settings(
                        console="auto",  # 自动捕获控制台输出
                        x_disable_service=True,  # 禁用服务
                     ))
    if run:
        print(f">>> W&B Initialized. Mode: {run.settings.mode}")
    else:
        print(">>> W&B Init failed.")
        sys.exit(1)
except Exception as e:
    print(f">>> W&B Init Exception: {e}")
    sys.exit(1)

print(">>> Starting to print logs...")
for i in range(5):
    print(f"This is log line {i}")
    wandb.log({"step": i}) # 同时记录指标
    time.sleep(0.5)

print(">>> Finished printing logs.")
# wandb.finish() # 通常会自动调用，但在简单脚本中显式调用无妨
print(">>> Script finished.")