import pandas as pd

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
ROUNDING_TOLERANCE_USD = 0.011


def clean_order_items(order_items: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = order_items.drop_duplicates().reset_index(drop=True)

    assert df.isnull().sum().sum() == 0
    assert (df["unit_price_usd"] > 0).all()
    assert (df["quantity"] > 0).all()
    assert (df["line_total_usd"] > 0).all()

    calculated_total = (df["unit_price_usd"] * df["quantity"]).round(2)
    assert (calculated_total - df["line_total_usd"]).abs().max() <= ROUNDING_TOLERANCE_USD

    assert df["product_id"].isin(products["product_id"]).all()
    assert df["order_id"].isin(orders["order_id"]).all()

    return df


def clean_orders(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()

    assert df.isnull().sum().sum() == 0
    assert not df.duplicated().any()
    assert (df["subtotal_usd"] > 0).all()
    assert (df["total_usd"] > 0).all()
    assert df["discount_pct"].between(0, 100).all()

    calculated_total = (df["subtotal_usd"] * (1 - df["discount_pct"] / 100)).round(2)
    assert (calculated_total - df["total_usd"]).abs().max() <= ROUNDING_TOLERANCE_USD

    assert df["customer_id"].isin(customers["customer_id"]).all()

    return df


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()

    assert df.isnull().sum().sum() == 0
    assert not df.duplicated().any()
    assert (df["price_usd"] > 0).all()
    assert (df["cost_usd"] > 0).all()
    assert (df["cost_usd"] <= df["price_usd"]).all()

    calculated_margin = (df["price_usd"] - df["cost_usd"]).round(2)
    assert (calculated_margin - df["margin_usd"]).abs().max() <= ROUNDING_TOLERANCE_USD

    return df


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    df = customers.copy()

    assert df.isnull().sum().sum() == 0
    assert not df.duplicated().any()
    assert df["age"].between(18, 100).all()
    assert df["email"].str.match(EMAIL_PATTERN).all()

    return df
