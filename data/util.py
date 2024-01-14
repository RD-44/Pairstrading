import sqlite3
import numpy as np 
import pandas as pd 

class OHLCV:
    def __init__(self, db_conn: sqlite3.Connection) -> None:
        self.conn = db_conn # expects a connection to PairsTradingData.db

    def load(self, pairs: list, exchanges: list, since = '2020-01-01 00:00:00') -> list[pd.DataFrame]:
        dfs = []
        columns = [
            'unix time',
            'timestamp',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'volume base',
            'tradecount'
        ]
        assert(len(pairs) == len(exchanges))
        csr = self.conn.cursor()

        ids = []
        for pair, exc in zip(pairs, exchanges):
            csr.execute(
                f'''SELECT id FROM Pairs WHERE symbol = ? AND exchange = ?;''',
                (pair, exc)
            )
            ids.append(csr.fetchone()[0])
        print(ids)
        for id in ids:
            if type(since) == str:
                csr.execute(
                    f'''
                    SELECT unix_time, datetime, open, high, low, close, volume_transacted, volume_base, tradecount 
                    FROM OHLCV_Data
                    WHERE pair_id = ? AND datetime >= ?;
                    ''',
                    (id, since)
                )
            
            elif type(since) == int:
                csr.execute(
                    f'''
                    SELECT unix_time, datetime, open, high, low, close, volume_transacted, volume_base, tradecount
                    FROM OHLCV_Data
                    WHERE pair_id = ? AND unix >= ?;,
                    ''',
                    (id, since)
                )
            
            dfs.append(
                pd.DataFrame(
                    data = csr.fetchall(),
                    columns = columns
                ).set_index(['unix time', 'timestamp'])
                )
        csr.close()
        return dfs
            

