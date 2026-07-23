from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .dynamic import DynamicDataset
from .dnerf import DNeRFDataset
from .hypernerf import HyperNerfDataset
from .dynerf import Neural3D_NDC_Dataset
from .iphone_temp_2 import IphoneDataset
from .immersive_temp_hyper_reel import ImmersiveDataset


dataset_dict = {'immersive':ImmersiveDataset,
                'dynamic':DynamicDataset,
                'iphone':IphoneDataset,
                'dnerf':DNeRFDataset,
                'hypernerf':HyperNerfDataset,
                'Dynerf':Neural3D_NDC_Dataset,
               'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
                'own_data':YourOwnDataset}