"""flsys: A Flower / PyTorch app."""

import torch
import json
from random import random
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigsRecord, Metrics
from flsys.task import Net, get_weights, load_data, set_weights, test, train
from flsys.models.MobileNetV3 import get_mobilenet_v3_small_model


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, context: Context):
        self.client_state = context.state
        self.id = context.node_id

        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
         # add a reference to the state of your ClientApp
        if "fit_metrics" not in self.client_state.configs_records:
            self.client_state.configs_records["fit_metrics"] = ConfigsRecord()


    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        #print(config)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config["lr"],
            self.device,
        )

        fit_metrics = self.client_state.configs_records["fit_metrics"]

        if "train_loss_hist" not in fit_metrics:
            fit_metrics["train_loss_hist"] = [train_loss]
        else:
            fit_metrics["train_loss_hist"].append(train_loss)

        complex_metric = {"a" : 123, "b" : random(), "mylist" : [1, 2, 3, 4]}
        metric_str = json.dumps(complex_metric)
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss
             ,"my_metric": metric_str},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device) 
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}
    

    


def client_fn(context: Context):
    # Load model and data
    #net = Net()
    net = get_mobilenet_v3_small_model(10,True)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, context).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
