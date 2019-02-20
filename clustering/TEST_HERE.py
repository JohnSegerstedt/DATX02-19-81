import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
df.rename(index=str, columns={"A": "a", "B": "c"})