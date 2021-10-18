import os
import time
import math
import torch
import random
import argparse
import cate_dataset
import cate_model
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from transformers import AdamW, get_linear_schedule_with_warmup

import warnings

warnings.filterwarnings(action='ignore')

# 전처리된 데이터가 저장된 디렉터리
DB_PATH = f'../input/processed'

# 토큰을 인덱스로 치환할 때 사용될 사전 파일이 저장된 디렉터리
VOCAB_DIR = os.path.join(DB_PATH, 'vocab')

# 학습된 모델의 파라미터가 저장될 디렉터리
MODEL_PATH = f'../model'


# 미리 정의된 설정 값
class CFG:
    learning_rate = 3.0e-4  # 러닝 레이트
    batch_size = 1024  # 배치 사이즈
    num_workers = 4  # 워커의 개수
    print_freq = 100  # 결과 출력 빈도
    start_epoch = 0  # 시작 에폭
    num_train_epochs = 10  # 학습할 에폭수
    warmup_steps = 100  # lr을 서서히 증가시킬 step 수
    max_grad_norm = 10  # 그래디언트 클리핑에 사용
    weight_decay = 0.01
    dropout = 0.2  # dropout 확률
    hidden_size = 512  # 은닉 크기
    intermediate_size = 256  # TRANSFORMER셀의 intermediate 크기
    nlayers = 2  # BERT의 층수
    nheads = 8  # BERT의 head 개수
    seq_len = 64  # 토큰의 최대 길이
    n_b_cls = 57 + 1  # 대카테고리 개수
    n_m_cls = 552 + 1  # 중카테고리 개수
    n_s_cls = 3190 + 1  # 소카테고리 개수
    n_d_cls = 404 + 1  # 세카테고리 개수
    vocab_size = 32000  # 토큰의 유니크 인덱스 개수
    img_feat_size = 2048  # 이미지 피처 벡터의 크기
    type_vocab_size = 30  # 타입의 유니크 인덱스 개수
    csv_path = os.path.join(DB_PATH, 'train.csv')
    h5_path = os.path.join(DB_PATH, 'train_img_feat.h5')


def main():
    pass


if __name__ == '__main__':
    main()
