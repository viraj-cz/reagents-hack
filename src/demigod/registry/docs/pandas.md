## pandas

Tabular data analysis. Available as `import pandas as pd`. `numpy` and
`pyarrow` are installed alongside it.

**Use it by writing and running Python scripts.** Write the script to a file in
your output directory, run it with `python`, and keep the file. The script is
part of your evidence: it is how someone re-runs your method.

Reading input:

```python
import pandas as pd

df = pd.read_csv("/run/shared/data.csv")  # also read_parquet, read_json, read_excel
```

Writing output — always into your own output directory:

```python
df.to_csv("/run/out/summary.csv", index=False)
df.to_parquet("/run/out/summary.parquet")  # pyarrow-backed
```

Traps that cost real time here:

- `/run/shared` is **read-only**. Writing there raises `OSError`. Copy first if
  you need to mutate.
- `read_csv` infers dtypes per-chunk and will silently make a mixed column
  `object`. If a column matters, pass `dtype=` explicitly.
- Chained assignment (`df[df.a > 1]['b'] = 0`) silently does nothing. Use
  `.loc[mask, 'b'] = 0`.
- `NaN != NaN`. Use `.isna()`, never `== np.nan`.
- Default `merge` is an inner join and will drop rows without warning. Pass
  `how=` explicitly and assert on `len(df)` after.
- Print `df.shape` and `df.dtypes` right after loading. Most wrong answers here
  come from silently loading the wrong number of rows.

When you state a number in your `claim`, the script that produced it must be in
`evidence`.
