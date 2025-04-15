import torch
import os
import json
import wandb
from datetime import datetime
from flwr.common import Parameters, FitRes, parameters_to_ndarrays, FitIns, Scalar, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx, FedAdam, FedAvgM, FedAdagrad, FedYogi
from flsys.task import Net, set_weights
from flsys.models.MobileNetV3 import get_mobilenet_v3_small_model




class CustomFedAvg(FedAvg):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.result_to_save = {}
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_param_save_dir = os.path.join("global_model_param",name)
        os.mkdir(self.model_param_save_dir)
        wandb.init(project="FL_image_recognition_sys", name=f"custom-strategy-FedAvg-{name}")

    def aggregate_fit(self, 
                      server_round: int, 
                      results: list[tuple[ClientProxy, FitRes]], 
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
                      ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        parameters_aggregated, metrics_aggregated =  super().aggregate_fit(server_round, results, failures)
        
        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        #model = Net()
        model = get_mobilenet_v3_small_model(10,False)
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated


    def evaluate(self, 
                 server_round:int, 
                 parameters: Parameters
                 ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        loss, metrics =  super().evaluate(server_round, parameters)
        
        my_results = {"loss" : loss, **metrics}
        self.result_to_save[server_round] = my_results

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)
        
        #log to WB
        wandb.log(my_results,step=server_round)

        return loss, metrics


class CustomFedProx(FedProx):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_to_save = {}
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_param_save_dir = os.path.join("global_model_param",name)
        os.mkdir(self.model_param_save_dir)
        wandb.init(project="FL_image_recognition_sys", name=f"custom-strategy-FedProx-{name}")
    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        #model = Net()
        model = get_mobilenet_v3_small_model(10,False)
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self, 
                 server_round:int, 
                 parameters: Parameters
                 ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        loss, metrics =  super().evaluate(server_round, parameters)
        
        my_results = {"loss" : loss, **metrics}
        self.result_to_save[server_round] = my_results

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)
        
        #log to WB
        wandb.log(my_results,step=server_round)

        return loss, metrics
    
    def configure_fit(self, server_round, parameters, client_manager) -> list[tuple[ClientProxy, FitIns]]:
        client_config_pairs =  super().configure_evaluate(server_round, parameters, client_manager)

        if server_round >= 5:
            self.proximal_mu = self.proximal_mu / 2.0
        

            return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "proximal_mu": self.proximal_mu},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]



class CustomFedAdam(FedAdam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_to_save = {}
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_param_save_dir = os.path.join("global_model_param",name)
        os.mkdir(self.model_param_save_dir)
        wandb.init(project="FL_image_recognition_sys", name=f"custom-strategy-FedAvg-{name}")

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        #model = Net()
        model = get_mobilenet_v3_small_model(10,False)
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self, server_round:int, parameters: Parameters) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        loss, metrics =  super().evaluate(server_round, parameters)

        my_results = {"loss" : loss, **metrics}
        self.result_to_save[server_round] = my_results

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4
                      )

        #log to WB
        wandb.log(my_results,step=server_round)

        return loss, metrics
    

class CustomFedAvgM(FedAvgM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_to_save = {}
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_param_save_dir = os.path.join("global_model_param",name)
        os.mkdir(self.model_param_save_dir)
        wandb.init(project="FL_image_recognition_sys", name=f"custom-strategy-FedAvgM-{name}")

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        #model = Net()
        model = get_mobilenet_v3_small_model(10,False)
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated

    def evaluate(self, server_round:int, parameters: Parameters) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        loss, metrics =  super().evaluate(server_round, parameters)

        my_results = {"loss" : loss, **metrics}
        self.result_to_save[server_round] = my_results

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4
                      )

        #log to WB
        wandb.log(my_results,step=server_round)

        return loss, metrics


class CustomFedAdagrad(FedAdagrad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_to_save = {}
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_param_save_dir = os.path.join("global_model_param",name)
        os.mkdir(self.model_param_save_dir)
        wandb.init(project="FL_image_recognition_sys", name=f"custom-strategy-FedAdagrad-{name}")

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        #model = Net()
        model = get_mobilenet_v3_small_model(10,False)
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated


    def evaluate(self, server_round:int, parameters: Parameters) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        loss, metrics =  super().evaluate(server_round, parameters)

        my_results = {"loss" : loss, **metrics}
        self.result_to_save[server_round] = my_results

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4
                      )

        #log to WB
        wandb.log(my_results,step=server_round)

        return loss, metrics


class CustomFedYogi(FedYogi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.result_to_save = {}
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_param_save_dir = os.path.join("global_model_param",name)
        os.mkdir(self.model_param_save_dir)
        wandb.init(project="FL_image_recognition_sys", name=f"custom-strategy-FedYogi-{name}")

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        #model = Net()
        model = get_mobilenet_v3_small_model(10,False)
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated


    def evaluate(self, server_round:int, parameters: Parameters) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        loss, metrics =  super().evaluate(server_round, parameters)

        my_results = {"loss" : loss, **metrics}
        self.result_to_save[server_round] = my_results

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4
                      )

        #log to WB
        wandb.log(my_results,step=server_round)

        return loss, metrics


def get_strategy(strategy_name:str, *args, **kwargs):
    if strategy_name == "FedAvg":
        return CustomFedAvg(*args, **kwargs)
    elif strategy_name == "FedProx":
        return CustomFedProx(*args, **kwargs)
    elif strategy_name == "FedAdam":
        return CustomFedAdam(*args, **kwargs)
    elif strategy_name == "FedAvgM":
        return CustomFedAvgM(*args, **kwargs)
    elif strategy_name == "FedAdagrad":
        return CustomFedAdagrad(*args, **kwargs)
    elif strategy_name == "FedYogi":
        return CustomFedYogi(*args, **kwargs)
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")
    




