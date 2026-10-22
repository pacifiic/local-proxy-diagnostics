import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest

if __package__:
    from .restore_artifacts import restore_one, safe_path
else:
    from restore_artifacts import restore_one, safe_path


class RestoreTests(unittest.TestCase):
    def fixture(self, root):
        content = b'item,value\n1,0.5\n'
        packed = gzip.compress(content, mtime=0)
        (root / 'data.csv.gz').write_bytes(packed)
        return content, dict(path='data.csv', compressed_path='data.csv.gz',
                             size=len(content), sha256=hashlib.sha256(content).hexdigest(),
                             compressed_sha256=hashlib.sha256(packed).hexdigest())

    def test_restore_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            content, record = self.fixture(root)
            self.assertEqual(restore_one(root, record), 'restored')
            self.assertEqual((root / 'data.csv').read_bytes(), content)
            self.assertEqual(restore_one(root, record), 'existing')

    def test_preserve_different_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, record = self.fixture(root)
            (root / 'data.csv').write_bytes(b'keep me')
            with self.assertRaises(FileExistsError):
                restore_one(root, record)
            self.assertEqual((root / 'data.csv').read_bytes(), b'keep me')

    def test_corruption_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, record = self.fixture(root)
            record['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                restore_one(root, record)
            self.assertFalse((root / 'data.csv').exists())
            self.assertFalse(list(root.glob('.restore-*')))

    def test_verify_without_materializing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, record = self.fixture(root)
            self.assertEqual(restore_one(root, record, True), 'verified')
            self.assertFalse((root / 'data.csv').exists())

    def test_reject_escaped_path(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                safe_path(Path(d), '../outside.csv')
            with self.assertRaises(ValueError):
                safe_path(Path(d), '/outside.csv')


if __name__ == '__main__':
    unittest.main()
