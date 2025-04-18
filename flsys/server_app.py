"""flsys: A Flower / PyTorch app."""
import json
from typing import List,Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, DifferentialPrivacyServerSideFixedClipping
from flsys.task import Net, get_weights, set_weights, test, get_transforms
from flsys.models.MobileNetV3 import get_mobilenet_v3_small_model
from torch.utils.data import DataLoader
from datasets import load_dataset
from flsys.my_strategy import CustomFedAvg


def get_evaluate_fn(testloader,device):
    """return a callback that evaluates the gloabl model"""
    def evaluate(server_round, parameters_ndarrays, config):
        #Instance model
        #net = Net()
        net = get_mobilenet_v3_small_model(10,False)
        #Apply global_model parameter
        set_weights(net,parameters_ndarrays)
        net.to(device)
        #Run test
        loss, accuarcy = test(net, testloader,device)

        return loss, {"cen_accuarcy":accuarcy}
    
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
    #ndarrays = get_weights(Net())
    ndarrays = get_weights(get_mobilenet_v3_small_model(10,True))
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


# Create ServerApp
app = ServerApp(server_fn=server_fn)


