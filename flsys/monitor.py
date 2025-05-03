import subprocess
import time
import os
import signal
import psutil
import threading
import sys
import re
import colorama

# colorama.init()

# ANSI转义序列过滤正则表达式
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# 定义输出流处理函数
def stream_output(stream, prefix=''):
    last_line = ""
    for line in iter(stream.readline, ''):
        if not line:
            break
            
        # 过滤ANSI控制字符
        clean_line = ansi_escape.sub('', line)
        
        # 防止重复输出相同内容
        if clean_line.strip() != last_line.strip():
            print(f"{prefix}{line}", end='')
            sys.stdout.flush()  # 确保立即显示
            last_line = clean_line
        
        # 检查是否是wandb完成同步的信号
        if "Synced" in line and "W&B file" in line:
            global wandb_completed
            wandb_completed = True

# 启用Windows ANSI支持
if os.name == 'nt':
    os.system('')  # 这会激活Windows控制台的ANSI处理

# 其余代码保持不变
wandb_completed = False

# 启动flower进程 - 添加PYTHONIOENCODING环境变量
my_env = os.environ.copy()
my_env["PYTHONIOENCODING"] = "utf-8"
process = subprocess.Popen(
    ["flwr", "run", "."], 
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,  # 行缓冲，确保实时输出
    env=my_env
)

# 保存进程ID便于后续操作
pid = process.pid
print(f"启动Flower训练，进程ID: {pid}")

# 创建两个线程分别处理stdout和stderr
stdout_thread = threading.Thread(target=stream_output, args=(process.stdout, ''))
stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, ''))

# 设置为守护线程，这样主程序退出时它们也会退出
stdout_thread.daemon = True
stderr_thread.daemon = True

# 启动线程
stdout_thread.start()
stderr_thread.start()

# 主循环等待wandb完成或进程结束
while not wandb_completed and process.poll() is None:
    time.sleep(0.5)

# 如果wandb已完成，等待额外时间让可能的清理工作完成
if wandb_completed:
    print("\n检测到wandb同步完成，等待5秒后终止进程...")
    time.sleep(5)

# 如果进程仍在运行，终止它和所有子进程
if process.poll() is None:
    print("正在终止进程树...")
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # 终止所有子进程
        for child in children:
            print(f"  终止子进程: {child.pid}")
            child.terminate()
        
        # 终止主进程
        print(f"  终止主进程: {pid}")
        parent.terminate()
        
        # 给进程一点时间自行终止
        gone, alive = psutil.wait_procs(children + [parent], timeout=3)
        
        # 如果还有进程存活，强制杀死
        for p in alive:
            print(f"  强制终止进程: {p.pid}")
            p.kill()
    except:
        # 最后尝试系统命令杀进程
        os.system(f"taskkill /F /PID {pid} /T")

print("训练完成，所有进程已终止")