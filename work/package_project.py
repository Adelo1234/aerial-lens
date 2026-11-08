from pathlib import Path
import zipfile

root = Path('outputs/drone-tracking')
with zipfile.ZipFile('outputs/drone-tracking.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            bundle.write(path, Path('drone-tracking')/path.relative_to(root))
print('Updated project ZIP')
