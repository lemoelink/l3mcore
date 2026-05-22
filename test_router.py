import os
import json
from modules.config_manager import ConfigManager
from modules.generic_router import GenericRouter

def test():
    config = ConfigManager()
    router = GenericRouter(config)
    
    if router.router_type == 'embedding':
        print("Model loaded successfully as embedding.")
        label, score = router.predict("Quiero ver mis correos electrónicos")
        print(f"Predicted label: {label}, Score: {score}")
        
        label, score = router.predict("Como crear una clase abstracta en Python")
        print(f"Predicted label: {label}, Score: {score}")

if __name__ == '__main__':
    test()
