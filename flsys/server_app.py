"""flsys: A Flower / PyTorch app."""
import json
import wandb
import os
import psutil
import signal
import atexit
import sys
import time
from typing import List,Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, Grid, LegacyContext
from flwr.server.strategy import FedAvg, DifferentialPrivacyServerSideFixedClipping
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flsys.task import Net, get_weights, set_weights, test, get_transforms
from models.MobileNetV3 import get_mobilenet_v3_small_model
from torch.utils.data import DataLoader
from datasets import load_dataset
from flsys.my_strategy import CustomFedAvg

from flsys.config import config as configLoader

def get_evaluate_fn(testloader,device):
    """return a callback that evaluates the gloabl model"""
    def evaluate(server_round, parameters_ndarrays, config):
        #Instance model
        if configLoader.model.type == "mobilenet_v3_small":
            net = get_mobilenet_v3_small_model(10,False)
        else:
            net = Net()

        #net = Net()
        #Apply global_model parameter
        set_weights(net,parameters_ndarrays)
        net.to(device)
        #Run test
        loss, accuracy = test(net, testloader,device)

        return loss, {"cen_accuracy":accuracy}
    
    return evaluate

def weighted_average(metrics: List[Tuple[int,Metrics]]) -> Metrics:
    """A function  that aggregates metrics"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    #return weighted average examples
    return {"accuarcy": sum(accuracies) / total_examples}

def handle_fit_metrics(metrics: List[Tuple[int,Metrics]]) -> Metrics:
    """handle metrics from fit method in clients"""
    b_values = []
    for _, m in metrics:
        my_metric_str = m["my_metric"]

        my_metric = json.loads(my_metric_str)
        b_values.append(my_metric["b"])

    return {"max_b" : max(b_values)}

def on_fit_config(server_round: int) -> Metrics:
    """Adjusts learing rate based on current round"""
    lr = 0.01
    if server_round > 2:
        lr = 0.005
    return {"lr": lr}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    #print("---------------------------",context.run_config)
    #num_clients = context.node_config["num-clients"]

    # Initialize model parameters
    if config.model.type == "mobilenet_v3_small":
        ndarrays = get_weights(get_mobilenet_v3_small_model(10,True))
    else:
        ndarrays = get_weights(Net())


    # ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # load test dataset
    testset = load_dataset("uoft-cs/cifar10")["test"]

    #construct testloader
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)

    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.5,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader,device="cpu"),
        fit_metrics_aggregation_fn=handle_fit_metrics,
    )

    dp_strategy = DifferentialPrivacyServerSideFixedClipping(
        strategy=strategy,
        noise_multiplier=0.7,
        clipping_norm=5.0,
        num_sampled_clients=10,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# 在main函数结束前或适当位置添加此函数
def terminate_all_processes():
    try:
        # 获取当前进程
        current_process = psutil.Process(os.getpid())
        
        # 获取父进程
        parent = current_process.parent()
        
        # 终止所有子进程和相关进程
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            # 避免终止重要的系统进程
            if 'python' in proc.info['name'].lower() and proc.info['pid'] != current_pid:
                # 检查命令行参数，确认是flwr相关进程
                if any('flwr' in cmd.lower() for cmd in proc.info['cmdline'] if cmd):
                    try:
                        proc.terminate()  # 尝试优雅终止
                        proc.wait(timeout=3)  # 等待进程终止
                    except:
                        try:
                            proc.kill()  # 如果优雅终止失败，强制终止
                        except:
                            pass  # 忽略任何错误
        
        # 最后终止自己
        os._exit(0)
    except:
        # 确保即使出错也会终止当前进程
        os._exit(0)


# Create ServerApp
app = ServerApp()
#app = ServerApp(server_fn=server_fn)

@app.main()
def main(grid:Grid, context:Context):

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    
    # Initialize model parameters
    # if configLoader.model.type == "mobilenet_v3_small":
    #     ndarrays = get_weights(get_mobilenet_v3_small_model(10,True))
    # else:
    #     ndarrays = get_weights(Net())

    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # load test dataset
    testset = load_dataset("uoft-cs/cifar10")["test"]

    #construct testloader
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)


    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader,device="cpu"),
        fit_metrics_aggregation_fn=handle_fit_metrics,
    )


    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


    fit_workflow = SecAggPlusWorkflow(
        num_shares=context.run_config["num-shares"],
        reconstruction_threshold=context.run_config["reconstruction-threshold"],
        max_weight=context.run_config["max-weight"],
    )


        
    # Create the workflow
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # Execute
    workflow(grid, context)
    try:
        wandb.finish()
    except:
        pass

  
    # 只终止与Flower相关的进程，而不是所有Python进程
    def terminate_flower_only():
        current_pid = os.getpid()
        print(f"准备终止Flower相关进程（当前进程ID: {current_pid}）")
        
        # 获取当前进程及其父进程，避免终止它们
        try:
            current = psutil.Process(current_pid)
            parent_pid = current.parent().pid
            print(f"当前进程的父进程ID: {parent_pid}（将被保留）")
            protected_pids = [parent_pid]  # 只保护父进程
        except:
            protected_pids = []
        
        processes_to_terminate = []
        # 先收集所有需要终止的进程，排除保护的进程
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # 跳过受保护的进程
                if proc.pid in protected_pids:
                    continue
                    
                # 只查找Python进程
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join([cmd for cmd in proc.info['cmdline'] if cmd])
                    # 更精确的匹配，避免误杀终端进程
                    if ('flwr' in cmdline.lower() or 
                        'flower' in cmdline.lower() or 
                        'server_app.py' in cmdline.lower() or
                        'client_app.py' in cmdline.lower()):
                        processes_to_terminate.append(proc)
            except:
                pass
        
        # 然后终止这些进程，包括当前进程自己
        for proc in processes_to_terminate:
            if proc.pid != current_pid:  # 先终止其他进程
                try:
                    print(f"终止进程: {proc.pid}")
                    proc.terminate()
                except:
                    pass
        
        print("相关进程已终止，正常退出当前进程")
        
        # 最后使用更强制的方法终止自己 - 替换sys.exit(0)
        # os._exit(0)  # 使用os._exit强制终止当前进程，不执行任何清理
        os.kill(current_pid,2)

    # 执行终止操作
    terminate_flower_only()

# 注册程序退出时的清理函数
# atexit.register(terminate_all_processes)

   