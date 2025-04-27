from flsys.config import config

if __name__ == "__main__":
    print(f"当前模型: {config.model.type}")
    print(f"W&B项目: {config.wandb.project}")
    print(type(config.model.type))