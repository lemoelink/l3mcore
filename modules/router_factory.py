"""
LEMoE Router Factory
--------------------
Instantiates the correct router based on config.json -> router.mode:

  "generic"  -> GenericRouter (user trained model + categories.jsonl)
  "model"    -> DecisionRouter (fine-tuned model with hardcoded labels, e.g. grape-route)
"""

from modules.logger import app_logger


def create_router(config_manager):
    """
    Reads router.mode from config and returns the correct instance.
    Both classes expose the same interface:
        predict(text) -> (label: str, score: float)
        clear_cache()
    """
    cfg = config_manager.get('router', {})
    mode = cfg.get('mode', 'generic').lower()

    if mode == 'model':
        from modules.decision_router import DecisionRouter
        app_logger.info("RouterFactory: 'model' mode (Pre-trained ML DecisionRouter)")
        return DecisionRouter(config_manager)
    else:
        from modules.generic_router import GenericRouter
        app_logger.info("RouterFactory: 'generic' mode (GenericRouter + categories.jsonl)")
        return GenericRouter(config_manager)
