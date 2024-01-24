import sqlite3
import numpy as np
from data import util
from evolutionary.nsga import NSGA


def nzc(series):
    return np.sum(np.where(np.diff(np.signbit(series)))[0])


def plot(functions):
    nsga = NSGA(functions)
    population = nsga.generate_population()
    print("Generation 0")
    for i in range(20):
        population = nsga.evolve(population)
        print("Generation ", i+1)
        print("Average norm: ", nsga.avg_objective(population, 0))
        print("Average standard deviation: ",
              nsga.avg_objective(population, 1))

    print(population)


# Create a SQLite database connection
db_conn = sqlite3.connect("PairsTradingData.db")

# Instantiate the OHLCV class with the database connection
ohlc_reader = util.OHLCV(db_conn)

# Call the load method with the required parameters
cryptos = ["LINKUSDT", "MATICUSDT"]
exchanges = ["binance" for _ in range(len(cryptos))]
since_date = "2024-01-01 00:01:00"

data_frames = ohlc_reader.load(cryptos, exchanges, since_date)

functions = [lambda x: np.linalg.norm(x[0]*data_frames[0]["close"] - x[1]*data_frames[1]["close"]),
             lambda x: -np.std(x[0]*data_frames[0]["close"] - x[1]*data_frames[1]["close"])]

# -np.std(x[0]*data_frames[0]["close"] - x[1]*data_frames[1]["close"])
plot(functions)
# test = list(combinations(assets, 2))
