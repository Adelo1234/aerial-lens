import tempfile
import unittest
import zipfile
from pathlib import Path
from drone_project.download import extract_archive, FOLDER


class ArchiveTests(unittest.TestCase):
    def test_expected_layout_extracts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root/'sample.zip'
            with zipfile.ZipFile(archive,'w') as bundle:
                bundle.writestr(f'{FOLDER}/images/sample.jpg', b'fixture')
                bundle.writestr(f'{FOLDER}/annotations/sample.txt', '0,0,1,1,1,1,0,0')
            target = extract_archive(archive,root/'data')
            self.assertTrue((target/'annotations/sample.txt').is_file())
            with self.assertRaises(ValueError):
                extract_archive(archive,root/'data')

    def test_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root/'unsafe.zip'
            with zipfile.ZipFile(archive,'w') as bundle:
                bundle.writestr('../outside.txt', 'bad')
            with self.assertRaisesRegex(ValueError,'Unsafe archive'):
                extract_archive(archive,root/'data')
            self.assertFalse((root/'outside.txt').exists())

    def test_wrong_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root/'wrong.zip'
            with zipfile.ZipFile(archive,'w') as bundle:
                bundle.writestr('README.txt','not the training archive')
            with self.assertRaisesRegex(ValueError,'Expected'):
                extract_archive(archive,root/'data')
            self.assertFalse((root/'data'/FOLDER).exists())
