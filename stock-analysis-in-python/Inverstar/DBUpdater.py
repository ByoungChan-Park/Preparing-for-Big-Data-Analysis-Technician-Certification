from datetime import datetime
from urllib.request import urlopen

import json
import pandas as pd
import pymysql
import ssl
from bs4 import BeautifulSoup


class DBUpdater:

    def __init__(self):
        self.conn = pymysql.connect(host='localhost', user='primavera', password='primavera', db='primavera', charset='utf8')
        self.context = ssl._create_unverified_context()

        with self.conn.cursor() as curs:
            sql = """
                CREATE TABLE IF NOT EXISTS COMPANY_INFO (
                    CODE VARCHAR(20),
                    COMPANY VARCHAR(40),
                    LAST_UPDATE DATE,
                    PRIMARY KEY (CODE)
                )
            """
            curs.execute(sql)

            sql = """
                CREATE TABLE IF NOT EXISTS DAILY_PRICE (
                    CODE VARCHAR(20),
                    REG_DATE DATE,
                    OPEN BIGINT(20),
                    HIGH BIGINT(20),
                    LOW BIGINT(20),
                    CLOSE BIGINT(20),
                    DIFF BIGINT(20),
                    VOLUME BIGINT(20),
                    PRIMARY KEY (CODE, REG_DATE)
                )
            """
            curs.execute(sql)

        self.conn.commit()
        self.codes = dict()
        self.update_comp_info()

    def __del__(self):
        self.conn.close()

    def read_krx_code(self):
        url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
        krx = pd.read_html(urlopen(url, context=context).read())[0]
        krx = krx[['종목코드', '회사명']]
        krx = krx.rename(columns={'종목코드': 'code', '회사명': 'company'})
        krx.code = krx.code.map('{:06d}'.format)
        return krx

    def update_comp_info(self):
        sql = "SELECT * FROM COMPANY_INFO"
        df = pd.read_sql(sql, self.conn)
        for idx in range(len(df)):
            self.codes[df['CODE'].values[idx]] = df['COMPANY'].values[idx]

        with self.conn.cursor() as curs:
            sql = 'SELECT MAX(LAST_UPDATE) FROM COMPANY_INFO'
            curs.execute(sql)
            rs = curs.fetchone()
            today = datetime.today().strftime('%Y-%m-%d')
            if rs[0] is None or rs[0].strftime('%Y-%m-%d') < today:
                krx = self.read_krx_code()
                for idx in range(len(krx)):
                    code = krx.code.values[idx]
                    company = krx.company.values[idx]
                    sql = f"REPLACE INTO COMPANY_INFO (CODE, COMPANY, LAST_UPDATE) VALUES ('{code}', '{company}', '{today}')"
                    curs.execute(sql)
                    self.codes[code] = company
                    tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                    print(f"[{tmnow}] {idx:04d} REPLACE INTO COMPANY_INFO VALUES ({code}, {company}, {today})")
                self.conn.commit()
                print('')

    def read_naver(self, code, company, pages_to_fetch):
        try:
            url = f"http://finance.naver.com/item/sise_day.nhn?code={code}"
            with urlopen(url, context=self.context) as doc:
                if doc is None:
                    return None
                html = BeautifulSoup(doc, 'lxml')
                pgrr = html.find("td", class_="pgRR")
                if pgrr is None:
                    return None
                s = str(pgrr.a["href"]).split('=')
                lastpage = s[-1]
            df = pd.DataFrame()
            pages = min(int(lastpage), pages_to_fetch)
            for page in range(1, pages):
                pg_url = '{}&page={}'.format(url, page)
                df = df.append(pd.read_html(urlopen(pg_url, context=self.context).read())[0])
                tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                print("[{}] {} ({}) {:04d}/{:04d} pages are downloading....".format(tmnow, company, code, page, pages), end="\r")
            df = df.rename(columns={'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})
            df['date'] = df['date'].replace('.', '-')
            df = df.dropna()
            df[['close', 'diff', 'open', 'high', 'low', 'volume']] = df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)
            df = df[['date', 'open', 'high', 'low', 'close', 'diff', 'volume']]
        except Exception as e:
            print('Exception occured :', str(e))
            return None
        return df

    def replace_into_db(self, df, num, code, company):
        with self.conn.cursor() as curs:
            for r in df.itertuples():
                sql = f"REPLACE INTO DAILY_PRICE VALUES ('{code}', '{r.date}', {r.open}, {r.high}, {r.low}, {r.close}, {r.diff}, {r.volume})"
                curs.execute(sql)
            self.conn.commit()
            print('[{}] #{:04d} {} ({}) : {} rows > REPLACE INTO DAILY_PRICE [OK]'.format(datetime.now().strftime('%Y-%m-%d %H:%M'), num + 1, company, code, len(df)))

    def update_daily_price(self, pages_to_fetch):
        for idx, code in enumerate(self.codes):
            df = self.read_naver(code, self.codes[code], pages_to_fetch)
            if df is None:
                continue
            self.replace_into_db(df, idx, code, self.codes[code])

    def execute_daily(self):
        self.update_comp_info()
        try:
            with open('config.json', 'r') as in_file:
                config = json.load(in_file)
                pages_to_fetch = config['pages_to_fetch']
        except FileNotFoundError:
            with open('config.json', 'w') as out_file:
                pages_to_fetch = 100
                config = {'pages_to_fetch': 1}
                json.dump(config, out_file)
        self.update_daily_price(pages_to_fetch)


if __name__ == '__main__':
    dbu = DBUpdater()
    dbu.execute_daily()
