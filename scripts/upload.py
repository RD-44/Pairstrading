import sqlite3
from sqlalchemy import create_engine
import numpy as np
import pandas as pd

# Uploads csv files from
# www.cryptodatadownload.com
# to a SQL database

def add_pair(conn, name, exchange):
    '''
    adds a pairname to the reference table if
    that pair and exchange doesnt exist in the table,
    otherwise does nothing.
    '''
    csr = conn.cursor()
    csr.execute(
        f'''
        SELECT id FROM Pairs WHERE symbol = '{name}' AND exchange = '{exchange}';
        '''
    )
    if not csr.fetchall():
        csr.execute(
            '''INSERT INTO Pairs (symbol, exchange) VALUES (?,?);''',
            (name, exchange)
        ) 
        conn.commit()
    csr.close()

def add_data_from_df(conn, pairname, exchange, df):
    '''
    Adds entire dataframe to sql table
    '''
    csr = conn.cursor()
    csr.execute(
            f'''
            SELECT id
            FROM Pairs
            WHERE symbol = '{pairname}' AND exchange = '{exchange}';
            '''
    )
    id = csr.fetchone()
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
            VALUES (?,?,?,?,?,?,?,?,?,?);
        ''',
        np.c_[np.full((df.shape[0],), id), df.to_numpy()].tolist()
    )
    conn.commit()
    csr.close()
    



if __name__ == '__main__':
    dbname = 'PairsTradingData.db'  # CHANGE AS NEEDED
    pairname = 'ETHUSDT'            # CHANGE AS NEEDED
    exchange = 'binance'            # CHANGE AS NEEDED

    # loading data into pandas
    # REPLACE YEARS WITH THE YEARS YOU HAVE CSV FILES FOR
    for i in ['2020', '2021', '2022', '2023', '2024']:
        fname = f'Binance_ETHUSDT_{i}_minute.csv' # CHANGE AS NEEDED
        df = pd.read_csv(
            fname,
            skiprows=0,
            header=1,
            low_memory=False
        )
        df.drop(labels=df.columns[2], axis='columns', inplace=True)

        # uploading to sql
        conn = sqlite3.connect(dbname)
        add_pair(conn, pairname, exchange)
        add_data_from_df(conn, pairname, exchange, df)
    conn.close() 

