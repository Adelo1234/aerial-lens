import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from drone_project.download_all import CATALOG, download_all, extract_one


class AllDownloadTests(unittest.TestCase):
    def test_catalog_includes_all_twenty_files(self):
        self.assertEqual(len(CATALOG), 20)
        self.assertEqual(len({row[2] for row in CATALOG}), 20)
        self.assertEqual({task: sum(row[0] == task for row in CATALOG)
                          for task in {row[0] for row in CATALOG}},
                         dict(DET=4, VID=4, SOT=7, MOT=4, CC=1))

    def test_complete_reuse_and_failed_link_retry(self):
        calls = []
        failing = {CATALOG[1][2]}
        def fake_download(id, output, **kwargs):
            calls.append(id)
            if id in failing: raise RuntimeError('Fixture quota error')
            with zipfile.ZipFile(output, 'w') as bundle:
                bundle.writestr('dataset/images/frame.txt', 'fixture')
            return output
        fake = types.SimpleNamespace(download=fake_download)
        space = types.SimpleNamespace(free=500*10**9)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, gdown=fake), \
                patch('drone_project.download_all.shutil.disk_usage', return_value=space), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete downloads'):
                download_all(directory)
            manifest = json.loads((Path(directory)/'download-all-manifest.json').read_text())
            self.assertEqual(sum(x['status'] == 'complete' for x in manifest.values()), 19)
            failing.clear()
            result = download_all(directory)
            self.assertTrue(all(x['status'] == 'complete' for x in result.values()))
            self.assertEqual(len(calls), 21)  # only failed file fetched on retry
            download_all(directory)
            self.assertEqual(len(calls), 21)

    def test_insufficient_disk_stops_before_network(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, gdown=types.SimpleNamespace()), \
                patch('drone_project.download_all.shutil.disk_usage', return_value=types.SimpleNamespace(free=10**9)), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, 'Not enough disk'):
                download_all(directory)

    def test_verified_drive_quota_is_not_local_cache_capacity(self):
        def fake_download(id, output, **kwargs):
            with zipfile.ZipFile(output, 'w') as bundle:
                bundle.writestr('data.txt', 'fixture')
            return output
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, gdown=types.SimpleNamespace(download=fake_download)), \
                patch('drone_project.download_all.shutil.disk_usage', return_value=types.SimpleNamespace(free=88*10**9)), \
                contextlib.redirect_stdout(io.StringIO()):
            result = download_all(directory, drive_free_gb=188.85)
            self.assertEqual(len(result), 20)
            self.assertTrue(all(item['status'] == 'complete' for item in result.values()))

    def test_extract_and_reject_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'fixture.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('part/data.txt', 'ok')
            target = extract_one(archive, root/'data')
            self.assertEqual((target/'part/data.txt').read_text(), 'ok')
            self.assertEqual(extract_one(archive, target), target)
            with zipfile.ZipFile(root/'unsafe.zip', 'w') as bundle:
                bundle.writestr('../escape.txt', 'bad')
            with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
                extract_one(root/'unsafe.zip', root/'unsafe')


if __name__ == '__main__': unittest.main()
