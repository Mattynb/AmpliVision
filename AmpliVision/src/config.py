from dataclasses import dataclass, field

@dataclass
class Config:
    
    TAG : str = ""
    use_case : str = ""

    # IMAGE
    dataset : str = ""
    path_to_imgs : str = "" 
    scanned_path : str = ""
    SIZE : list = field(default_factory=lambda: [1024, 1024]) # image size for CNN input
    
    # TRAINING
    TARGETS : list = field(default_factory=list)
    model_name : str = "LENET"
    EPOCHS : int = 48
    BATCH_N : int = 16
    STEPS_PER_EPOCH : int = 50
    VALIDATION_STEPS : int = 20
    BLACK: bool = False #if generated images will show only the painted tests area(making everything else black) or not

    def initialize(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def display(self):
        print(f"\n{'*'*25} CONFIG {'*'*25}")
        for field_ in self.__dataclass_fields__:
            print(f"{field_} = \"{getattr(self, field_)}\"")
        print(f"{'*'*58}\n")

CONFIG = Config()