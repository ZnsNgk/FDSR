'''
Import your models here and join them into model_list
'''
from .FDSR_woDGM import FDSR_woDGM
from .FDSR_wStep import FDSR_wStep
from .FDSR_4conv import FDSR as FDSR_new
from .FDSR_RK2 import FDSR as FDSR_RK2
from .FDSR_RK2_DeepS import FDSR_DS as FDSR_RK2_DS
from .DeFiAN import Generator as DeFiAN
from .MSRN import MSRN
from .OISR import OISR
from .CARN import CARN
from .FDSR_pre import FDSR_pre


model_list = {
    "FDSR_pre": FDSR_pre,
    "FDSR_woDGM": FDSR_woDGM,
    "FDSR_wStep": FDSR_wStep,
    "FDSR_1": FDSR_4conv,
    "FDSR_2": FDSR_4conv,
    "FDSR_4": FDSR_4conv,
    "FDSR_8": FDSR_4conv,
    "FDSR_16": FDSR_4conv,
    "FDSR_32": FDSR_4conv,
    "FDSR_RK2": FDSR_RK2,
    "FDSR_RK2_1": FDSR_RK2,
    "FDSR_RK2_2": FDSR_RK2,
    "FDSR_RK2_4": FDSR_RK2,
    "FDSR_RK2_16": FDSR_RK2,
    "FDSR_RK2_32": FDSR_RK2,
    "FDSR_RK2_64": FDSR_RK2,
    "FDSR_RK2_32_DS": FDSR_RK2_DS,
    "DeFiAN": DeFiAN,
    "MSRN": MSRN,
    "OISR": OISR,
    "CARN": CARN,
}

def get_model(model_name, **kwargs):
    return model_list[model_name](**kwargs)