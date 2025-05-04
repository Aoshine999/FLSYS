import torch
import os
import json
import wandb
from datetime import datetime
from flwr.common import Parameters, FitRes, parameters_to_ndarrays, FitIns, Scalar, NDArrays, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx, FedAdam, FedAvgM, FedAdagrad, FedYogi
from flsys.task import Net, set_weights
from models.MobileNetV3 import get_mobilenet_v3_small_model
from flsys.config import config as configLoader



class CustomFedAvg(FedAvg):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.result_to_save = {}
        self.model_name = configLoader.model.type

        self.pj_stDate = datetime.now()

        self.pj_edDate = None


        name = self.model_name + "-custom-strategy-FedAvg-" + self.pj_stDate.strftime("%Y-%m-%d_%H-%M-%S")
        
        self.model_param_save_dir = os.path.join("global_model_param",name)
        self.result_to_save["model_name"] = name
        self.result_to_save["total_train_time"] = None

        os.mkdir(self.model_param_save_dir)

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)

        wandb.init(project="FL_image_recognition_sys", name=f"{name}", id=f"{self.model_name}-custom-strategy-FedAvg")

    def aggregate_fit(self, 
                      server_round: int, 
                      results: list[tuple[ClientProxy, FitRes]], 
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
                      ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        parameters_aggregated, metrics_aggregated =  super().aggregate_fit(server_round, results, failures)
        
        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
        #model = Net()
  
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        #torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated


    def evaluate(self, 
                 server_round:int, 
                 parameters: Parameters
                 ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        
        best_acc = 0.0
        loss, metrics =  super().evaluate(server_round, parameters)
        
        my_results = {"loss" : loss, **metrics}



        #存储最优模型
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
            
        #model = Net()    
        set_weights(model,parameters_ndarrays)

        if metrics["cen_accuracy"] > best_acc:
            best_acc = metrics["cen_accuracy"]
            torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,"best_model.pth"))




        self.result_to_save[server_round] = my_results

        self.pj_edDate = datetime.now()

        # 计算并记录训练时间
        hours = int ((self.pj_edDate - self.pj_stDate).total_seconds() // 3600)  # 转换为小时
        minutes = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 3600 // 60)  # 转换为分钟
        seconds = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 60)  # 转换为秒
        total_time = f"{hours}h {minutes}m {seconds}s"

        self.result_to_save["total_train_time"] = total_time


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
        self.model_name = configLoader.model.type

        self.pj_stDate = datetime.now()

        self.pj_edDate = None


        name = self.model_name + "-custom-strategy-FedProx-" + self.pj_stDate.strftime("%Y-%m-%d_%H-%M-%S")
        
        self.model_param_save_dir = os.path.join("global_model_param",name)
        self.result_to_save["model_name"] = name
        self.result_to_save["total_train_time"] = None

        os.mkdir(self.model_param_save_dir)

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)

        wandb.init(project="FL_image_recognition_sys", name=f"{name}", id=f"{self.model_name}-custom-strategy-FedProx")

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated =  super().aggregate_fit(server_round, results, failures)
        
        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
        #model = Net()
  
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        #torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self, 
                 server_round:int, 
                 parameters: Parameters
                 ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        best_acc = 0.0
        loss, metrics =  super().evaluate(server_round, parameters)
        
        my_results = {"loss" : loss, **metrics}



        #存储最优模型
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
            
        #model = Net()    
        set_weights(model,parameters_ndarrays)

        if metrics["cen_accuracy"] > best_acc:
            best_acc = metrics["cen_accuracy"]
            torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,"best_model.pth"))




        self.result_to_save[server_round] = my_results

        self.pj_edDate = datetime.now()

        # 计算并记录训练时间
        hours = int ((self.pj_edDate - self.pj_stDate).total_seconds() // 3600)  # 转换为小时
        minutes = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 3600 // 60)  # 转换为分钟
        seconds = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 60)  # 转换为秒
        total_time = f"{hours}h {minutes}m {seconds}s"

        self.result_to_save["total_train_time"] = total_time


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
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        self.eta = 0.01
        self.eta_l = 0.01
        self.tau = 0.001

        
        self.result_to_save = {}
        self.model_name = configLoader.model.type

        self.pj_stDate = datetime.now()

        self.pj_edDate = None


        name = self.model_name + "-custom-strategy-FedAdam-" + self.pj_stDate.strftime("%Y-%m-%d_%H-%M-%S")
        
        self.model_param_save_dir = os.path.join("global_model_param",name)
        self.result_to_save["model_name"] = name
        self.result_to_save["total_train_time"] = None

        os.mkdir(self.model_param_save_dir)

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)
        
        wandb.init(project="FL_image_recognition_sys", name=f"{name}", id=f"{self.model_name}-custom-strategy-FedAdam")
    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated =  super().aggregate_fit(server_round, results, failures)
        
        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
        #model = Net()
  
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        #torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self, server_round:int, parameters: Parameters) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        best_acc = 0.0
        loss, metrics =  super().evaluate(server_round, parameters)
        
        my_results = {"loss" : loss, **metrics}



        #存储最优模型
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
            
        #model = Net()    
        set_weights(model,parameters_ndarrays)

        if metrics["cen_accuracy"] > best_acc:
            best_acc = metrics["cen_accuracy"]
            torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,"best_model.pth"))




        self.result_to_save[server_round] = my_results

        self.pj_edDate = datetime.now()

        # 计算并记录训练时间
        hours = int ((self.pj_edDate - self.pj_stDate).total_seconds() // 3600)  # 转换为小时
        minutes = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 3600 // 60)  # 转换为分钟
        seconds = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 60)  # 转换为秒
        total_time = f"{hours}h {minutes}m {seconds}s"

        self.result_to_save["total_train_time"] = total_time


        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)
        
        #log to WB
        wandb.log(my_results,step=server_round)

        return loss, metrics
    

class CustomFedAvgM(FedAvgM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.result_to_save = {}
        self.model_name = configLoader.model.type

        self.pj_stDate = datetime.now()

        self.pj_edDate = None


        name = self.model_name + "-custom-strategy-FedAvgM-" + self.pj_stDate.strftime("%Y-%m-%d_%H-%M-%S")
        
        self.model_param_save_dir = os.path.join("global_model_param",name)
        self.result_to_save["model_name"] = name
        self.result_to_save["total_train_time"] = None

        os.mkdir(self.model_param_save_dir)

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)

        wandb.init(project="FL_image_recognition_sys", name=f"{name}", id=f"{self.model_name}-custom-strategy-FedAvgM")

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated =  super().aggregate_fit(server_round, results, failures)
        
        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
        #model = Net()
  
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        #torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated

    def evaluate(self, server_round:int, parameters: Parameters) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        best_acc = 0.0
        loss, metrics =  super().evaluate(server_round, parameters)
        
        my_results = {"loss" : loss, **metrics}



        #存储最优模型
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
            
        #model = Net()    
        set_weights(model,parameters_ndarrays)

        if metrics["cen_accuracy"] > best_acc:
            best_acc = metrics["cen_accuracy"]
            torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,"best_model.pth"))




        self.result_to_save[server_round] = my_results

        self.pj_edDate = datetime.now()

        # 计算并记录训练时间
        hours = int ((self.pj_edDate - self.pj_stDate).total_seconds() // 3600)  # 转换为小时
        minutes = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 3600 // 60)  # 转换为分钟
        seconds = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 60)  # 转换为秒
        total_time = f"{hours}h {minutes}m {seconds}s"

        self.result_to_save["total_train_time"] = total_time


        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)
        
        #log to WB
        wandb.log(my_results,step=server_round)

        return loss, metrics


class CustomFedAdagrad(FedAdagrad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.result_to_save = {}
        self.model_name = configLoader.model.type

        self.pj_stDate = datetime.now()

        self.pj_edDate = None


        name = self.model_name + "-custom-strategy-FedAdagrad-" + self.pj_stDate.strftime("%Y-%m-%d_%H-%M-%S")
        
        self.model_param_save_dir = os.path.join("global_model_param",name)
        self.result_to_save["model_name"] = name
        self.result_to_save["total_train_time"] = None

        os.mkdir(self.model_param_save_dir)

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)

        wandb.init(project="FL_image_recognition_sys", name=f"{name}", id=f"{self.model_name}-custom-strategy-FedAdagrad")


    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated =  super().aggregate_fit(server_round, results, failures)
        
        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
        #model = Net()
  
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        #torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated


    def evaluate(self, server_round:int, parameters: Parameters) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        best_acc = 0.0
        loss, metrics =  super().evaluate(server_round, parameters)
        
        my_results = {"loss" : loss, **metrics}



        #存储最优模型
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
            
        #model = Net()    
        set_weights(model,parameters_ndarrays)

        if metrics["cen_accuracy"] > best_acc:
            best_acc = metrics["cen_accuracy"]
            torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,"best_model.pth"))




        self.result_to_save[server_round] = my_results

        self.pj_edDate = datetime.now()

        # 计算并记录训练时间
        hours = int ((self.pj_edDate - self.pj_stDate).total_seconds() // 3600)  # 转换为小时
        minutes = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 3600 // 60)  # 转换为分钟
        seconds = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 60)  # 转换为秒
        total_time = f"{hours}h {minutes}m {seconds}s"

        self.result_to_save["total_train_time"] = total_time


        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)
        
        #log to WB
        wandb.log(my_results,step=server_round)

        return loss, metrics


class CustomFedYogi(FedYogi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.result_to_save = {}
        self.model_name = configLoader.model.type

        self.pj_stDate = datetime.now()

        self.pj_edDate = None


        name = self.model_name + "-custom-strategy-FedYogi-" + self.pj_stDate.strftime("%Y-%m-%d_%H-%M-%S")
        
        self.model_param_save_dir = os.path.join("global_model_param",name)
        self.result_to_save["model_name"] = name
        self.result_to_save["total_train_time"] = None

        os.mkdir(self.model_param_save_dir)

        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)

        wandb.init(project="FL_image_recognition_sys", name=f"{name}", id=f"{self.model_name}-custom-strategy-FedYogi")


    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures: list[tuple[ClientProxy, FitRes] | BaseException]) -> tuple[Parameters | None, dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated =  super().aggregate_fit(server_round, results, failures)
        
        # convert parameters to ndarrys
        ndarrays  = parameters_to_ndarrays(parameters_aggregated)

        #instance the model
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
        #model = Net()
  
        set_weights(model,ndarrays)

        #save model parameters in the standard pytorch way
        #torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,f"model_round_{server_round}"))

        return parameters_aggregated, metrics_aggregated


    def evaluate(self, server_round:int, parameters: Parameters) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        best_acc = 0.0
        loss, metrics =  super().evaluate(server_round, parameters)
        
        my_results = {"loss" : loss, **metrics}



        #存储最优模型
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        if configLoader.model.type == "MobileNetV3":
            model = get_mobilenet_v3_small_model(10,False)
        else:
            model = Net()
            
        #model = Net()    
        set_weights(model,parameters_ndarrays)

        if metrics["cen_accuracy"] > best_acc:
            best_acc = metrics["cen_accuracy"]
            torch.save(model.state_dict(), os.path.join(self.model_param_save_dir,"best_model.pth"))




        self.result_to_save[server_round] = my_results

        self.pj_edDate = datetime.now()

        # 计算并记录训练时间
        hours = int ((self.pj_edDate - self.pj_stDate).total_seconds() // 3600)  # 转换为小时
        minutes = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 3600 // 60)  # 转换为分钟
        seconds = int ((self.pj_edDate - self.pj_stDate).total_seconds() % 60)  # 转换为秒
        total_time = f"{hours}h {minutes}m {seconds}s"

        self.result_to_save["total_train_time"] = total_time


        json_file_path = os.path.join(self.model_param_save_dir,"result.json")
        #save result to json file
        with open(json_file_path,"w") as json_file:
            json.dump(self.result_to_save,json_file,indent=4)
        
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
    




