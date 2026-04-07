from dataclasses import dataclass, field
import datetime
import os

@dataclass
class Config:
    
    DateTime : str = datetime.datetime.now().strftime('%Y/%m/%d %H:%M')
    TAG : str = ""
    SAVE_NAME : str = ""
    use_case : str = ""

    # IMAGE
    dataset : str = ""
    path_to_imgs : str = "" 
    scanned_path : str = ""
    SIZE : list = field(default_factory=lambda: [512, 512]) # image size for CNN input
    SAVE : bool = False # if true make sure TRAIN_DATASET is set to GEN
    path_to_store : str = "/hpcstor6/scratch01/m/matheus.berbet001/" # f"{os.getcwd()}/AmpliVision/data/generated_images"  
    CROP_TO_TEST_AREA: bool = True
    
    MODEL_PARAMS : dict = ""

    # TRAINING
    TARGETS : list = field(default_factory=list)
    model_name : str = "LENET"
    EPOCHS : int = 2
    BATCH_N : int = 64
    STEPS_PER_EPOCH : int = 150
    VALIDATION_STEPS : int = 42 
    BLACK: bool = False #if generated images will show only the painted tests area(making everything else black) or not
    GEN_IMG_FORM : str = "tensor" # 'tensor' or 'numpy' for generated images format
    TRAIN_DATASET : str = "LOAD" # GEN or LOAD

    # TESTING
    TEST_DATASET : str = ""

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

if __name__ == "__main__":
    print(f"""{CONFIG.SIZE[0]}s{CONFIG.EPOCHS}e{CONFIG.BATCH_N}b{CONFIG.STEPS_PER_EPOCH}ts{CONFIG.VALIDATION_STEPS}vs_{CONFIG.TRAIN_DATASET}tr""")