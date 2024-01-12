import sqlite3
from sqlalchemy import create_engine
import numpy as np
import pandas as pd

# Uploads csv files from
# www.cryptodatadownload.com
# to a SQL database

def add_pair(conn, name, exchange):
    '''
    adds a pairname to the reference table
    '''
    csr = conn.cursor()
    csr.execute(
        '''INSERT OR IGNORE INTO Pairs (symbol, exchange) VALUES (?,?)''',
        (name, exchange)
    )
    conn.commit()
    csr.close()

def add_data_from_df(conn, pairname, exchange, df):
    '''
    Adds entire dataframe to sql table
    '''
    csr = conn.cursor()
    id = csr.execute(
            f'''
            SELECT id
            FROM Pairs
            WHERE symbol = '{pairname}' AND exchange = '{exchange}';
            '''
        )
    csr.executemany(
        '''
        INSERT INTO OHLCV_Data (
            pair_id,
            unix_time,
            datetime,
            open,
            high,
            low,
            close,
            volume_transacted,
            volume_base,
            tradecount
            )
            VALUES (?,?,?,?,?,?,?,?,?,?)
        ''',
        (np.c_[np.full((df.shape[0], 1), id), df.to_numpy()])
    )
    csr.commit()
    csr.close()
    



if __name__ == '__main__':
    dbname = 'PairsTradingData.db'  # CHANGE AS NEEDED
    pairname = 'ETHUSDT'            # CHANGE AS NEEDED
    exchange = 'binance'            # CHANGE AS NEEDED

    # loading data into pandas
    fname = 'Binance_ETHUSDT_2020_minute.csv' # CHANGE AS NEEDED
    df = pd.read_csv(
        fname,
        skiprows=0,
        header=1,
        low_memory=False
    )
    df['date'] = pd.to_datetime(df['date']) # make sure it is a datetime type
    df.drop(labels='symbol', axis='columns', inplace=True)
    df.insert(0, 'pair_id', np.full((df.shape[0],), 1), True)
    
    # uploading to sql
    conn = sqlite3.connect(dbname)
    add_pair(conn, pairname, exchange)
    id = conn.execute(
            f'''
            SELECT id
            FROM Pairs
            WHERE symbol = '{pairname}' AND exchange = '{exchange}';
            '''
        )
    engine = create_engine(f'sqlite:///{dbname}')
    df.to_sql('OHLCV_Data', con=engine, index=False, if_exists='replace')
    conn.close()
