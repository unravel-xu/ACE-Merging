from src.steps.step0 import run_step0
from src.steps.step1 import run_step1
from src.steps.step2 import run_step2
from src.steps.step3 import run_step3
# from src.steps.step4 import run_step4

class PipelineController:
    def __init__(self, config):
        self.config = config

    def run_all_steps(self):
        next_step = self.run_step(0)
        while(next_step):
            next_step = self.run_step(next_step)

    def run_step(self, step_number):
        runners = {
            0: run_step0,
            1: run_step1,
            2: run_step2,
            3: run_step3,
        }

        if step_number in runners:
            print(f"[STEP{step_number}] Running...")
            next_step = runners[step_number](self.config)
            print(f"[STEP{step_number}] Done\n")
            return next_step
        else:
            raise ValueError(f"Unknown step number: {step_number}")