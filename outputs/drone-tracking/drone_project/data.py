"""Validate and convert supplied VisDrone images; keep a reproducible split manifest."""
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
         'tricycle', 'awning-tricycle', 'bus', 'motor']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def raw_to_yolo(text, width, height):
    """VisDrone xywh pixels -> YOLO center xywh; ignore categories 0/11 and score 0."""
    result, counts = [], Counter()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(x.strip()) for x in line.strip().rstrip(',').split(',')]
            if len(values) < 8 or not all(math.isfinite(x) for x in values):
                raise ValueError('expected eight finite comma-separated values')
            x, y, w, h, score, category, _, _ = values[:8]
            if category != int(category):
                raise ValueError('category must be an integer')
            if score == 0 or category in (0, 11):
                counts['ignored'] += 1
                continue
            if w <= 0 or h <= 0:
                counts['invalid_box_size'] += 1
                continue
            if score < 0 or not 1 <= category <= 10:
                raise ValueError('invalid score, category, or box size')
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                counts['outside_image'] += 1
                continue
            if (x1, y1, x2, y2) != (x, y, x+w, y+h):
                counts['clipped'] += 1
            result.append(f'{int(category)-1} {(x1+x2)/2/width:.8f} '
                          f'{(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} '
                          f'{(y2-y1)/height:.8f}')
        except ValueError as exc:
            raise ValueError(f'annotation line {number}: {exc}') from exc
    return result, counts


def validate_yolo(text):
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError('expected class cx cy w h')
            c, x, y, w, h = map(float, fields)
            if not all(math.isfinite(v) for v in (c, x, y, w, h)):
                raise ValueError('non-finite value')
            if c != int(c) or not 0 <= c < 10:
                raise ValueError('class must be in 0..9')
            if not 0 <= x <= 1 or not 0 <= y <= 1 or not 0 < w <= 1 or not 0 < h <= 1:
                raise ValueError('invalid normalized coordinates')
            if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
                raise ValueError('box extends beyond image')
            result.append(f'{int(c)} {x:.8f} {y:.8f} {w:.8f} {h:.8f}')
        except ValueError as exc:
            raise ValueError(f'YOLO line {number}: {exc}') from exc
    return result


def prepare(images, labels, output, label_format='visdrone', seed=42, groups_csv=None):
    images, labels, output = map(lambda p: Path(p).resolve(), (images, labels, output))
    if not images.is_dir() or not labels.is_dir():
        raise ValueError('Images and labels must be existing directories.')
    if output.exists():
        raise ValueError('Output already exists. Choose a new directory to preserve prior data.')
    if output == images or images in output.parents or output == labels or labels in output.parents:
        raise ValueError('Output must be outside the source directories.')
    sources = sorted(p for p in images.rglob('*') if p.suffix.lower() in EXTENSIONS)
    if len(sources) < 10:
        raise ValueError('At least 10 images are required for a useful three-way split.')
    stems = [p.stem for p in sources]
    if len(set(stems)) != len(stems):
        raise ValueError('Duplicate image stems: flatten/rename images and matching labels first.')
    group_map = {}
    if groups_csv:
        with open(groups_csv, newline='', encoding='utf-8-sig') as stream:
            group_map = {r['image_stem']: r['group_id'] for r in csv.DictReader(stream)}
        if any(s not in group_map or not group_map[s] for s in stems):
            raise ValueError('Group CSV must cover every image_stem with a nonempty group_id.')
    records, counts = [], Counter()
    seen_hashes = {}
    # Validate everything BEFORE creating output. Duplicate images must share a group.
    for source in sources:
        label = labels / f'{source.stem}.txt'
        if not label.is_file():
            raise ValueError(f'Missing annotation: {label}. Use an empty file for background images.')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in seen_hashes:
            prior = seen_hashes[digest]
            if not groups_csv or group_map[source.stem] != group_map[Path(prior).stem]:
                raise ValueError(f'Duplicate image content: {source.name} and {prior}')
            counts['duplicate_images_grouped'] += 1
        seen_hashes[digest] = source.name
        with Image.open(source) as im:
            im.load()
            width, height = im.size
        text = label.read_text(encoding='utf-8-sig')
        try:
            if label_format == 'visdrone':
                rows, stats = raw_to_yolo(text, width, height)
                counts.update(stats)
            else:
                rows = validate_yolo(text)
        except ValueError as exc:
            raise ValueError(f'{label}: {exc}') from exc
        validate_yolo('\n'.join(rows))
        counts['boxes'] += len(rows)
        counts['background_images'] += not rows
        records.append(dict(source=source, rows=rows, sha256=digest,
                            group=group_map.get(source.stem, source.stem), width=width, height=height))
    grouped = defaultdict(list)
    for record in records:
        grouped[record['group']].append(record)
    keys = sorted(grouped)
    if len(keys) < 3:
        raise ValueError('At least three independent groups are needed.')
    random.Random(seed).shuffle(keys)
    # Image-level split is exactly rounded 70/20/10. Group-level split approximates it.
    n = len(keys)
    n_train = max(1, min(n-2, round(n*.7)))
    n_val = max(1, min(n-n_train-1, round(n*.2)))
    assignments = {k: ('train' if i < n_train else 'val' if i < n_train+n_val else 'test')
                   for i, k in enumerate(keys)}
    output.mkdir(parents=True)
    manifest, class_counts = [], {s: Counter() for s in ('train', 'val', 'test')}
    for record in records:
        split = assignments[record['group']]
        source = record['source']
        image_dir, label_dir = output/'images'/split, output/'labels'/split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_dir/source.name)
        (label_dir/f'{source.stem}.txt').write_text('\n'.join(record['rows'])+'\n', encoding='utf-8')
        for row in record['rows']:
            class_counts[split][NAMES[int(row.split()[0])]] += 1
        manifest.append(dict(image=source.name, split=split, group_id=record['group'],
                             sha256=record['sha256'], width=record['width'], height=record['height'],
                             objects=len(record['rows'])))
    config = {'path': output.as_posix(), 'train': 'images/train', 'val': 'images/val',
              'test': 'images/test', 'names': dict(enumerate(NAMES))}
    # JSON is a valid YAML subset and avoids a preprocessing dependency on PyYAML.
    (output/'data.yaml').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    summary = dict(seed=seed, label_format=label_format, grouped=bool(groups_csv),
                   images_by_split=dict(Counter(r['split'] for r in manifest)),
                   class_counts=class_counts, counts=counts,
                   split_note='Group ratios may differ from image ratios.' if groups_csv else
                   'Image-level split; supply groups.csv to keep related sequence frames together.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary

