import torch
import os
import json
import wandb
from datetime import datetime
from flwr.common import Parameters, FitRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx, FedAdam
from flsys.task import Net, set_weights, get_mobilenet_v3_small_model

class CustomFedAvg(FedAvg):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.result_to_save = {}
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_param_save_dir = os.path.join("global_model_param",name)
        os.mkdir(self.model_param_save_dir)
        wandb.init(project="FL_image_recognition_sys", name=f"custom-strategy-{name}")

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
    
