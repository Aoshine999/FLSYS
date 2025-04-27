import tomli
from pathlib import Path
from types import SimpleNamespace

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.load_config()
        return cls._instance
    
    def load_config(self, path: str = None):
        config_path = self._find_config_path()
        with open(config_path, "rb") as f:
            config_data = tomli.load(f)
        self._create_namespaces(config_data)
    
    def _create_namespaces(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleNamespace(**value))
            else:
                setattr(self, key, value)
    
    def _find_config_path(self):
        paths_to_try = [
            Path("config.toml"),
            Path(__file__).parent.parent / "config.toml"
        ]
        for path in paths_to_try:
            if path.exists():
                return path
        raise FileNotFoundError("找不到配置文件 config.toml")

# 初始化配置（项目启动时调用一次）
# config = Config()
