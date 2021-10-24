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

DB_PATH = f'../input/processed'

VOCAB_DIR = os.path.join(DB_PATH, 'vocab')

MODEL_PATH = f'../model'


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
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--nepochs", type=int, default=CFG.num_train_epochs)
    parser.add_argument("--seq_len", type=int, default=CFG.seq_len)
    parser.add_argument("--nworkers", type=int, default=CFG.num_workers)
    parser.add_argument("--wsteps", type=int, default=CFG.warmup_steps)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nlayers", type=int, default=CFG.nlayers)
    parser.add_argument("--nheads", type=int, default=CFG.nheads)
    parser.add_argument("--hidden_size", type=int, default=CFG.hidden_size)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--stratified", action='store_true')
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)
    args = parser.parse_args()
    print(args)

    # 키워드 인자로 받은 값을 CFG로 다시 저장합니다.
    CFG.batch_size = args.batch_size
    CFG.num_train_epochs = args.nepochs
    CFG.seq_len = args.seq_len
    CFG.num_workers = args.nworkers
    CFG.warmup_steps = args.wsteps
    CFG.learning_rate = args.lr
    CFG.dropout = args.dropout
    CFG.seed = args.seed
    CFG.nlayers = args.nlayers
    CFG.nheads = args.nheads
    CFG.hidden_size = args.hidden_size
    print(CFG.__dict__)

    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ['PYTHONHASHSEED'] = str(CFG.seed)
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    torch.cuda.manual_seed(CFG.seed)
    torch.backends.cudnn.deterministic = True

    # 전처리된 데이터를 읽어옵니다.
    print('loading ...')
    train_df = pd.read_csv(CFG.csv_path, dtype={'tokens': str})
    train_df['img_idx'] = train_df.index  # 몇 번째 행인지 img_idx 칼럼에 기록
    print(train_df.shape())


if __name__ == '__main__':
    main()
