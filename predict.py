import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Tuple


def _get_consts() -> Tuple[pd.Series, pd.Series, list, list]:
    MIN = pd.Series(
        data={
            'Bracing_age': 10.000000,
            'Before.bracing.Cobb': 25.000000,
            'C_DAR': 2.100000,
            'Best.In_brace.Correction': 100/9,
            'Twoyears.follow_up.Cobb': 16.000000,
        },
        dtype=np.float64
    )

    MAX = pd.Series(
        data={
            'Bracing_age': 15.00,
            'Before.bracing.Cobb': 45.00,
            'C_DAR': 11.25,
            'Best.In_brace.Correction': 100.00,
            'Twoyears.follow_up.Cobb': 60.00,
        },
        dtype=np.float64
    )

    COLS_1 = [
        'Gender',
        'Bracing_age',
        'Before.bracing.Cobb',
        'C_DAR',
        'Best.In_brace.Correction',
        'Bracing.Risser.sign_0',
        'Bracing.Risser.sign_1',
        'Bracing.Risser.sign_2',
        'Twoyears.follow_up.Cobb',
        'Outcome',
    ]

    COLS_2 = [
        'Bracing_age',
        'Before.bracing.Cobb',
        'C_DAR',
        'Best.In_brace.Correction',
        'Twoyears.follow_up.Cobb',
    ]

    return MIN, MAX, COLS_1, COLS_2


def _preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    MIN, MAX, COLS_1, COLS_2 = _get_consts()

    # One-hot encoding Bracing Risser Sign
    data = pd.get_dummies(
        raw_data,
        prefix='Bracing.Risser.sign',
        columns=['Bracing.Risser.sign']
    )
    data = data[COLS_1]

    # Mapping Gender to {0, 1}
    data['Gender'] = data['Gender'] - 1

    # Normalizing other features and Two Years Follow-up Cobb target
    data[COLS_2] = (data[COLS_2] - MIN) / (MAX - MIN)

    # Convert dataframe to numpy array
    X = data.iloc[:, :-2].to_numpy()

    return X


def _post_proccess(y_reg: np.ndarray) -> np.ndarray:
    MIN, MAX, _, _ = _get_consts()
    y_reg_pp = (y_reg*(MAX['Twoyears.follow_up.Cobb']-MIN['Twoyears.follow_up.Cobb']) + MIN['Twoyears.follow_up.Cobb']).reshape((-1,))
    return y_reg_pp


def test(xlsx_path: str, model_path: str = 'reg_model') -> Tuple[np.ndarray, np.ndarray]:
    X = _preprocess(pd.read_excel(xlsx_path).iloc[:, 1:])
    model = tf.keras.models.load_model(model_path)
    y_reg = _post_proccess(model.predict(X))
    y_cls = np.where(y_reg > 50, 1, 0)

    return y_reg, y_cls
