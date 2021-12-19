from .finetuner import FineTuner
from .vanilla_trainer import VanillaTrainer

TRAINERS = {
    'finetuner': FineTuner,
    'vanilla': VanillaTrainer
}
