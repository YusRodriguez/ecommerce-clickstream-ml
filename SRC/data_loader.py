from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data" / "Raw"


def load_order_items(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "order_items.csv")


def load_orders(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "orders.csv")


def load_products(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "products.csv")


def load_customers(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "customers.csv")
