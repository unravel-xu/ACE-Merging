from control.config import ConfigManager
from control.pipeline import PipelineController

def main():
    config = ConfigManager()
    controller = PipelineController(config)
    controller.run_all_steps()

if __name__ == "__main__":
    main()