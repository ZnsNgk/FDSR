'''
Import your models here and join them into model_list
'''

from .DeFiAN import Generator as DeFiAN
from .MSRN import MSRN
from .OISR import OISR
from .FDSR_HesRFA import FDSR as FDSR_HesRFA


model_list = {
    "DeFiAN": DeFiAN,
    "MSRN": MSRN,
    "OISR": OISR,
    "FDSR_HesRFA": FDSR_HesRFA
}

def get_model(model_name, **kwargs):
    return model_list[model_name](**kwargs)