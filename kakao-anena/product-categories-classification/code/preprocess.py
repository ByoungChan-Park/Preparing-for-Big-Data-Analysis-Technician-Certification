import os
import re
import h5py
import logging
import numpy as np
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm

RAW_DATA_DIR = "../input/raw_data"  # 카카오에서 다운로드 받은 데이터의 디렉터리
PROCESSED_DATA_DIR = '../input/processed'  # 전처리된 데이터가 저장될 디렉터리
VOCAB_DIR = os.path.join(PROCESSED_DATA_DIR, 'vocab')  # 전처리에 사용될 사전 파일이 저장될 디렉터리

# 학습에 사용될 파일 리스트
TRAIN_FILE_LIST = [
    "train.chunk.01",
    "train.chunk.02",
    "train.chunk.03",
    "train.chunk.04",
    "train.chunk.05",
    "train.chunk.06",
    "train.chunk.07",
    "train.chunk.08",
    "train.chunk.09"
]

# 개발에 사용될 파일 리스트. 공개 리더보드 점수를 내는데 사용된다.
DEV_FILE_LIST = [
    "dev.chunk.01"
]

# 테스트에 사용될 파일 리스트. 파이널 리더보드 점수를 내는데 사용된다.
TEST_FILE_LIST = [
    "test.chunk.01",
    "test.chunk.02",
]


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()


def remove_special_characters(sentence, lower=True):
    p = re.compile('[\!@#$%\^&\*\(\)\-\=\[\]\{\}\.,/\?~\+\'"|_:;><`┃]')
    sentence = p.sub(' ', sentence)
    sentence = ' '.join(sentence.split())
    if lower:
        sentence = sentence.lower()
    return sentence


def get_column_data(path_list, div, col):
    col_data = []
    for path in path_list:
        h = h5py.File(path, 'r')
        col_data.append(h[div][col][:])
        h.close()
    return np.concatenate(col_data)


def get_dataframe(path_list, div):
    pids = get_column_data(path_list, div, col='pid')
    products = get_column_data(path_list, div, col='product')
    bcates = get_column_data(path_list, div, col='bcateid')
    mcates = get_column_data(path_list, div, col='mcateid')
    scates = get_column_data(path_list, div, col='scateid')
    dcates = get_column_data(path_list, div, col='dcateid')

    df = pd.DataFrame({'pid': pids, 'product': products, 'bcateid': bcates, 'mcateid': mcates, 'scateid': scates, 'dcateid': dcates})

    df['pid'] = df['pid'].map(lambda x: x.decode('utf-8'))
    df['product'] = df['product'].map(lambda x: x.decode('utf-8'))
    return df


def train_spm(txt_path, spm_path, vocab_size=32000, input_sentence_size=1000000):
    spm.SentencePieceTrainer.Train(
        f' --input={txt_path} --model_type=bpe'
        f' --model_prefix={spm_path} --vocab_size={vocab_size}'
        f' --input_sentence_size={input_sentence_size}'
        f' --shuffle_input_sentence=true'
        f' --minloglevel=2'
    )


def save_column_data(input_path_list, div, col, n_img_rows, output_path):
    h_out = h5py.File(output_path, 'w')
    h_out.create_dataset(col, (n_img_rows, 2048), dtype=np.float32)

    offset_out = 0

    for in_path in tqdm(input_path_list, desc=f'{div},{col}'):
        h_in = h5py.File(in_path, 'r')
        sz = h_in[div][col].shape[0]
        h_out[col][offset_out:offset_out + sz] = h_in[div][col][:]
        offset_out += sz
        h_in.close()
    h_out.close()


def preprocess():
    train_path_list = [os.path.join(RAW_DATA_DIR, fn) for fn in TRAIN_FILE_LIST]
    dev_path_list = [os.path.join(RAW_DATA_DIR, fn) for fn in DEV_FILE_LIST]
    test_path_list = [os.path.join(RAW_DATA_DIR, fn) for fn in TEST_FILE_LIST]

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(VOCAB_DIR, exist_ok=True)

    logger.info('loading ...')
    train_df = get_dataframe(train_path_list, 'train')
    dev_df = get_dataframe(dev_path_list, 'dev')
    test_df = get_dataframe(test_path_list, 'test')

    train_df['product'] = train_df['product'].map(remove_special_characters)

    with open(os.path.join(VOCAB_DIR, 'product.txt'), 'w', encoding='utf-8') as f:
        f.write(train_df['product'].str.cat(sep='\n'))

    logger.info('training sentencepiece model ...')
    train_spm(txt_path=os.path.join(VOCAB_DIR, 'product.txt'), spm_path=os.path.join(VOCAB_DIR, 'spm'))  # spm 접두어

    os.remove(os.path.join(VOCAB_DIR, 'product.txt'))

    for dirname, _, filenames in os.walk(VOCAB_DIR):
        for filename in filenames:
            logger.info(os.path.join(dirname, filename))

    logger.info('tokenizing product ...')
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(VOCAB_DIR, 'spm.model'))

    train_df['tokens'] = train_df['product'].map(lambda x: " ".join(sp.EncodeAsPieces(x)))

    dev_df['product'] = dev_df['product'].map(remove_special_characters)
    dev_df['tokens'] = dev_df['product'].map(lambda x: " ".join([str(token_id) for token_id in sp.EncodeAsPieces(x)]))

    test_df['product'] = test_df['product'].map(remove_special_characters)
    test_df['tokens'] = test_df['product'].map(lambda x: " ".join([str(token_id) for token_id in sp.EncodeAsPieces(x)]))

    columns = ['pid', 'tokens', 'bcateid', 'mcateid', 'scateid', 'dcateid']
    train_df = train_df[columns]
    dev_df = dev_df[columns]
    test_df = test_df[columns]

    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'), index=False)

    logger.info('processing img_feat ...')
    save_column_data(train_path_list, div='train', col='img_feat', n_img_rows=len(train_df), output_path=os.path.join(PROCESSED_DATA_DIR, 'train_img_feat.h5'))
    save_column_data(dev_path_list, div='dev', col='img_feat', n_img_rows=len(dev_df), output_path=os.path.join(PROCESSED_DATA_DIR, 'dev_img_feat.h5'))
    save_column_data(test_path_list, div='test', col='img_feat', n_img_rows=len(test_df), output_path=os.path.join(PROCESSED_DATA_DIR, 'test_img_feat.h5'))

    for dirname, _, filenames in os.walk(PROCESSED_DATA_DIR):
        for filename in filenames:
            logger.info(os.path.join(dirname, filename))


if __name__ == '__main__':
    preprocess()
