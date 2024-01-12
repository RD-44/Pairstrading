import sqlite3

# Creates a local sqlite3 database for the storage of 1min tick crypto OHLCV data
# Database created in whichever folder you run the script

GLOBAL_DB_NAME = 'PairsTradingData.db'

def main():
    conn = sqlite3.connect(GLOBAL_DB_NAME)
    csr = conn.cursor()

    csr.execute(
        '''
    Create table Pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL
    );
    '''
    )

    csr.execute(
        '''
    CREATE TABLE OHLCV_Data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id INTEGER NOT NULL,
    unix_time INTEGER NOT NULL,
    datetime DATETIME NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume_transacted REAL,
    volume_base REAL,
    tradecount INTEGER,
    FOREIGN KEY (pair_id) REFERENCES Pairs (id)
    );
    '''
    )

    conn.commit()
    conn.close()


if __name__ == '__main__':
    main()