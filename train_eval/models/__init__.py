'''
Import your models here and join them into model_list
'''
from .FDSR_woDGM import FDSR_woDGM
from .FDSR_wStep import FDSR_wStep
from .FDSR_4conv import FDSR as FDSR_4conv
from .FDSR_RK2 import FDSR as FDSR_RK2
from .DeFiAN import Generator as DeFiAN
from .MSRN import MSRN
from .OISR import OISR
from .CARN import CARN
from .FDSR_pre import FDSR_pre
from .FDSR_DeFiAM import FDSR as FDSR_DeFiAM
from .FDSR_HesRFA import FDSR as FDSR_HesRFA


model_list = {
    "FDSR_woDGM": FDSR_woDGM,
    "FDSR_wStep": FDSR_wStep,
    "FDSR": FDSR_4conv,
    "FDSR_RK2": FDSR_RK2,
    "DeFiAN": DeFiAN,
    "MSRN": MSRN,
    "OISR": OISR,
    "CARN": CARN,
    "FDSR_HesRFA": FDSR_HesRFA,
    "FDSR_DeFiAM": FDSR_DeFiAM
}

def get_model(model_name, **kwargs):
    return model_list[model_name](**kwargs)