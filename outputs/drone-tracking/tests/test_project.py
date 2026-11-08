import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from drone_project.control import Detection, IBVSController, TargetLock
from drone_project.data import prepare, raw_to_yolo, validate_yolo


class ProjectTests(unittest.TestCase):
    def test_conversion_clipping_and_ignored_classes(self):
        rows, stats = raw_to_yolo('10,20,20,40,1,1,0,0\n0,0,10,10,0,3,0,0\n'
                                 '0,0,10,10,1,11,0,0\n-5,0,15,10,1,4,0,0',100,100)
        self.assertEqual(rows[0], '0 0.20000000 0.40000000 0.20000000 0.40000000')
        self.assertEqual(rows[1], '3 0.05000000 0.05000000 0.10000000 0.10000000')
        self.assertEqual(stats['ignored'],2)
        self.assertEqual(stats['clipped'],1)
        self.assertEqual(raw_to_yolo('10,20,20,40,1,1,0,0,',100,100)[0], rows[:1])
        invalid_rows, invalid_stats = raw_to_yolo('10,20,4,0,1,4,0,0',100,100)
        self.assertEqual(invalid_rows, [])
        self.assertEqual(invalid_stats['invalid_box_size'], 1)

    def test_invalid_labels_fail_loudly(self):
        for row in ('0 .5 .5 0 .2', '10 .5 .5 .2 .2', '0 nan .5 .2 .2', '0 .01 .5 .8 .2'):
            with self.assertRaises(ValueError):
                validate_yolo(row)
        with self.assertRaises(ValueError):
            raw_to_yolo('broken label',100,100)

    def fixture(self, root):
        images, labels = root/'images',root/'labels'
        images.mkdir(); labels.mkdir()
        for i in range(20):
            Image.new('RGB',(100,100),(i*10,20,50)).save(images/f'{i}.png')
            (labels/f'{i}.txt').write_text('10,10,20,20,1,4,0,0')
        return images,labels

    def test_reproducible_split_and_no_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images,labels = self.fixture(root)
            first = prepare(images,labels,root/'first')
            prepare(images,labels,root/'second')
            a = json.loads((root/'first/manifest.json').read_text())
            b = json.loads((root/'second/manifest.json').read_text())
            self.assertEqual(a,b)
            self.assertEqual(first['images_by_split'],dict(train=14,val=4,test=2))
            self.assertEqual(len({x['sha256'] for x in a}),20)
            with self.assertRaises(ValueError):
                prepare(images,labels,root/'first')

    def test_groups_do_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images,labels = self.fixture(root)
            groups = root/'groups.csv'
            groups.write_text('image_stem,group_id\n'+'\n'.join(f'{i},sequence-{i//2}' for i in range(20)))
            prepare(images,labels,root/'data',groups_csv=groups)
            records = json.loads((root/'data/manifest.json').read_text())
            seen = {}
            for record in records:
                self.assertEqual(seen.setdefault(record['group_id'],record['split']),record['split'])

    def test_duplicate_images_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images,labels = self.fixture(root)
            (images/'1.png').write_bytes((images/'0.png').read_bytes())
            with self.assertRaisesRegex(ValueError,'Duplicate image content'):
                prepare(images,labels,root/'data')
            self.assertFalse((root/'data').exists())

    def test_duplicate_images_allowed_only_in_same_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images, labels = self.fixture(root)
            (images/'1.png').write_bytes((images/'0.png').read_bytes())
            groups = root/'groups.csv'
            groups.write_text('image_stem,group_id\n'+'\n'.join(f'{i},{0 if i == 1 else i}' for i in range(20)))
            result = prepare(images, labels, root/'data', groups_csv=groups)
            records = json.loads((root/'data/manifest.json').read_text())
            duplicates = [r for r in records if r['image'] in ('0.png', '1.png')]
            self.assertEqual(len({r['split'] for r in duplicates}), 1)
            self.assertEqual(result['counts']['duplicate_images_grouped'], 1)
            groups.write_text('image_stem,group_id\n'+'\n'.join(f'{i},{i}' for i in range(20)))
            with self.assertRaisesRegex(ValueError, 'Duplicate image content'):
                prepare(images, labels, root/'unsafe', groups_csv=groups)

    def test_lost_target_never_switches_identity(self):
        lock = TargetLock((0,))
        a = Detection(1,0,.9,(0,0,10,10))
        b = Detection(2,0,.9,(0,0,20,20))
        self.assertEqual(lock.select([a,b]).track_id,2)
        self.assertIsNone(lock.select([a]))
        self.assertEqual(lock.select([b]).track_id,2)

    def test_controller_sign_limits_deadband_and_loss(self):
        c = IBVSController()
        self.assertEqual(c.command((40,40,60,60),100,100)['yaw'],0)
        cmd = c.command((80,80,100,100),100,100)
        self.assertEqual(cmd['yaw'],.6)
        self.assertEqual(cmd['vertical'],-.4)
        lost = c.command(None,100,100)
        self.assertEqual((lost['yaw'],lost['vertical'],lost['forward']),(0,0,0))

    def test_static_target_error_converges(self):
        c = IBVSController()
        ex,ey = .6,-.4
        initial = ex*ex+ey*ey
        for _ in range(160):
            cx,cy = 50*(1+ex),50*(1+ey)
            cmd = c.command((cx-5,cy-5,cx+5,cy+5),100,100)
            ex -= cmd['yaw']*.05
            ey += cmd['vertical']*.05
        self.assertLess(ex*ex+ey*ey,initial*.01)


if __name__ == '__main__':
    unittest.main()
