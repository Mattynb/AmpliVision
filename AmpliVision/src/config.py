from dataclasses import dataclass, field
import datetime
import os


MAX = 2
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_NUM_INTEROP_THREADS'] = f'{MAX}'
os.environ['TF_NUM_INTRAOP_THREADS'] = f'{MAX}'
os.environ['OMP_NUM_THREADS'] = f'{MAX}'
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(MAX)
tf.config.threading.set_intra_op_parallelism_threads(MAX)

@dataclass
class Config:
    MAX_THREADS = MAX

    DateTime : str = datetime.datetime.now().strftime('%Y/%m/%d %H:%M')
    TAG : str = ""
    SAVE_NAME : str = ""
    use_case : str = ""

    # --- CLASSIFIER IMAGE PARAMS ---
    dataset : str = ""
    path_to_imgs : str = "" 
    scanned_path : str = ""
    SIZE : list = field(default_factory=lambda: [256, 256]) # image size for CNN input
    SAVE : bool = False
    NOISE: float = 0.1 # percentage 0.01 - 1.00
    path_to_store : str = "/hpcstor6/scratch01/m/matheus.berbet001/" # f"{os.getcwd()}/AmpliVision/data/generated_images"  
    CROP_TO_TEST_AREA: bool = True #True

    # --- CLASSIFIER TRAINING PARAMS ---
    MODEL_PARAMS : dict = ""
    TARGETS : list = field(default_factory=lambda: ['breast', 'control', 'lung', 'ovarian', 'prostate', 'skin', 'thyroid'])
    model_name : str = "InceptionResNetV2"
    EPOCHS : int = 25
    BATCH_N : int = 16 #64 # power of 2. multiple of targets.len() for balanced
    STEPS_PER_EPOCH : int = 1 # size(dataset) / batch
    VALIDATION_STEPS : int = 4 #2 
    BLACK: bool = False #if generated images will show only the painted tests area(making everything else black) or not
    GEN_IMG_FORM : str = "tensor" # 'tensor' or 'numpy' for generated images format
    TRAIN_DATASET : str = "LOAD" # GEN or LOAD

    # --- CYCLEGAN PARAMS ---
    GAN_ON : bool = False
    GAN_SIZE : list = field(default_factory=lambda: [256, 256]) # Keep small for memory
    GAN_BATCH_N : int = 1                                       # 1 or 2 max for CycleGAN
    GAN_EPOCHS : int = 100
    GAN_STEPS_PER_EPOCH : int = 500
    gan_path_real : str = f"{os.getcwd()}/AmpliVision/data/scanned_MARKER"
    gan_path_synth : str = "/hpcstor6/scratch01/m/matheus.berbet001/clean/"
    GAN_SAVE_PATH : str = f"{os.getcwd()}/AmpliVision/data/ML_models/cyclegan_gen_synth_to_real.keras"
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