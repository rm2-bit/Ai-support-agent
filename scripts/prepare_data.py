from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW, PROCESSED
from src.conversations import load, direct_pairs

PROCESSED.mkdir(parents=True, exist_ok=True)
df=load(RAW)
pairs=direct_pairs(df)
out=PROCESSED/'direct_pairs.csv'
pairs.to_csv(out,index=False)
print(f'Wrote {len(pairs):,} direct customer→brand pairs to {out}')
print(pairs.head(10).to_string(index=False))
