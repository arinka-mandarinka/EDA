import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    df['Revenue'] = df['Revenue'].astype(int)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:

    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    return df


def save_preprocessed(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # 1. Загружаем данные
    data = load_data("online_shoppers_intention.csv")

    # 2. Заполняем пропуски
    data = fill_missing_values(data)

    # 3. Готовим целевую переменную
    data = prepare_target(data)

    # 4. Кодируем категориальные признаки
    data = encode_categorical(data)

    # 5. Сохраняем
    save_preprocessed(data, "preprocessed_data.csv")

    print("EDA завершён. preprocessed_data.csv создан.")
