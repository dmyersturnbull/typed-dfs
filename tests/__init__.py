from pathlib import Path
import inspect

import pandas as pd


def tmpfile() -> Path:
    caller = inspect.stack()[1][3]
    path = Path(__file__).parent.parent.parent / "resources" / "tmp" / (str(caller) + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sample_data():
    return [
        pd.Series({"abc": 1, "123": 2, "xyz": 3}),
        pd.Series({"abc": 4, "123": 5, "xyz": 6}),
    ]
