'''
Import your models here and join them into model_list
'''

from .FDSR_RK2_DeepS import FDSR_DS as FDSR_RK2_DS
from .DeFiAN import Generator as DeFiAN
from .MSRN import MSRN
from .OISR import OISR


model_list = {
    "FDSR_RK2_32_DS": FDSR_RK2_DS,
    "DeFiAN": DeFiAN,
    "MSRN": MSRN,
    "OISR": OISR,
}

def get_model(model_name, **kwargs):
    return model_list[model_name](**kwargs)