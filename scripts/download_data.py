from pathlib import Path
import shutil
import kagglehub

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'data' / 'raw'
out.mkdir(parents=True, exist_ok=True)
path = Path(kagglehub.dataset_download('thoughtvector/customer-support-on-twitter'))
print('Downloaded to:', path)
for p in path.rglob('*'):
    if p.is_file():
        target = out / p.name
        if not target.exists():
            shutil.copy2(p, target)
        print(target)
