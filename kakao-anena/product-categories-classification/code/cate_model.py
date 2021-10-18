import torch # 파이토치 패키지 임포트
import torch.nn as nn
from transformers import BertConfig, BertModel


class CateClassfier(nn.Module):
    """상품
    """

    def __init__(self, cfg):
        """
        매개변수
        cfg: hidden_size, nlayers 등 설정값을 가지고 있는 함수
        """