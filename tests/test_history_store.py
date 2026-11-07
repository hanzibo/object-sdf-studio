"""Destructive checks are confined to TemporaryDirectory, never real user data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from history_store import Catalog,accounting,CATEGORIES
from data_lock import DataAccess


class HistoryChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='studio-history-tests-')
        self.base=Path(self.temp.name)
        self.root=self.base/'studio';self.root.mkdir()
        self.dataset=self.root/'datasets/object_20261102_120000'
        (self.dataset/'rgb').mkdir(parents=True)
        (self.dataset/'depth').mkdir()
        (self.dataset/'masks').mkdir()
        (self.dataset/'cam_K.txt').write_text('1 0 0\n0 1 0\n0 0 1\n')
        self.file=self.dataset/'rgb/frame.png';self.file.write_bytes(b'X'*(2*1024*1024))
        (self.dataset/'depth/frame.png').write_bytes(b'depth')
        (self.dataset/'masks/frame.png').write_bytes(b'mask')
        self.export=self.root/'exports/dataset_20261102_120100'
        (self.export/'segment_001/rgb').mkdir(parents=True)
        os.link(self.file,self.export/'segment_001/rgb/frame.png')
        (self.export/'export_report.json').write_text(json.dumps(dict(dataset=str(self.dataset))))

    def tearDown(self):self.temp.cleanup()

    def test_hardlink_accounting_and_shared_copy_survives(self):
        catalog=Catalog(self.root).scan()
        nodes=[n for n in catalog.entries[self.dataset].nodes if n.path==self.file]
        shared=[n for n in catalog.entries[self.export].nodes if n.path.name=='frame.png']
        self.assertEqual(accounting(nodes)['reclaim'],0)
        self.assertEqual(accounting(nodes+shared)['reclaim'],self.file.stat().st_blocks*512)
        self.assertEqual(accounting(nodes+shared)['logical'],self.file.stat().st_size*2)
        masks=self.dataset/'masks'
        self.assertEqual(catalog.selected([self.dataset,masks])['items'],1)
        self.assertEqual(catalog.selected([self.dataset,masks])['logical'],catalog.selected([self.dataset])['logical'])
        report=catalog.delete([self.dataset,masks])
        self.assertEqual(report['deleted'],[str(self.dataset)])
        self.assertEqual((self.export/'segment_001/rgb/frame.png').read_bytes(),b'X'*(2*1024*1024))
        remaining=Catalog(self.root).scan()
        self.assertGreaterEqual(remaining.selected([self.export])['reclaim'],2*1024*1024)
        remaining.delete([self.export]);self.assertFalse(self.export.exists())

    def test_delete_both_links_once(self):
        catalog=Catalog(self.root).scan()
        totals=catalog.selected([self.dataset,self.export])
        self.assertGreaterEqual(totals['reclaim'],2*1024*1024)
        report=catalog.delete([self.dataset,self.export])
        self.assertEqual(len(report['deleted']),2)
        self.assertEqual(report['failed'],[])
        self.assertEqual(report['observed_free_delta'],sum(x['after']-x['before'] for x in report['volume_changes']))
        self.assertTrue((self.root/'runtime/history_deletions.jsonl').is_file())

    def test_symlink_does_not_delete_target(self):
        target=self.base/'precious';target.mkdir();(target/'keep.txt').write_text('keep')
        (self.dataset/'linked').symlink_to(target,target_is_directory=True)
        (self.root/'datasets/alias').symlink_to(self.dataset,target_is_directory=True)
        catalog=Catalog(self.root).scan()
        self.assertNotIn(self.root/'datasets/alias',catalog.entries)
        self.assertFalse(any(n.path==target/'keep.txt' for n in catalog.entries[self.dataset].nodes))
        catalog.delete([self.dataset])
        self.assertEqual((target/'keep.txt').read_text(),'keep')

    def test_replaced_directory_refused(self):
        catalog=Catalog(self.root).scan()
        self.dataset.rename(self.dataset.with_name('original'))
        self.dataset.mkdir()
        with self.assertRaisesRegex(ValueError,'替换'):catalog.delete([self.dataset])
        self.assertTrue(self.dataset.exists())

    def test_unindexed_and_root_deletion_refused(self):
        catalog=Catalog(self.root).scan()
        for target in [self.root,self.base,self.root/'datasets']:
            with self.assertRaises(ValueError):catalog.delete([target])

    def test_mount_boundary_refused(self):
        extra=self.dataset/'mount';extra.mkdir()
        real=os.path.ismount
        with patch('history_store.os.path.ismount',side_effect=lambda p:Path(p)==extra or real(p)):
            catalog=Catalog(self.root).scan()
            self.assertTrue(catalog.entries[self.dataset].errors)
            with self.assertRaises(ValueError):catalog.delete([self.dataset])

    def test_active_access_lock_and_crash_release(self):
        catalog=Catalog(self.root).scan()
        with DataAccess(runtime=self.root/'runtime'):
            with self.assertRaisesRegex(RuntimeError,'正在采集'):catalog.delete([self.dataset])
        with DataAccess(exclusive=True,runtime=self.root/'runtime'):
            with self.assertRaises(RuntimeError):
                with DataAccess(runtime=self.root/'runtime'):pass
        code='from pathlib import Path; from data_lock import DataAccess; import os; lease=DataAccess(runtime=Path(__import__("sys").argv[1]));lease.__enter__();os._exit(0)'
        subprocess.run([sys.executable,'-c',code,str(self.root/'runtime')],cwd=Path(__file__).resolve().parents[1],check=True)
        catalog.delete([self.dataset]);self.assertFalse(self.dataset.exists())

    def test_active_legacy_cli_refused(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)','worker.py',str(self.dataset)])
        try:
            catalog=Catalog(self.root).scan()
            self.assertTrue(catalog.entries[self.dataset].active)
            with self.assertRaisesRegex(ValueError,'正被任务使用'):catalog.delete([self.dataset])
        finally:child.terminate();child.wait()

    def test_partial_failure_reported(self):
        catalog=Catalog(self.root).scan()
        import shutil
        remove=shutil.rmtree
        def simulate(path,*args,**kwargs):
            if Path(path)==self.export:raise PermissionError('test failure')
            return remove(path,*args,**kwargs)
        with patch('history_store.shutil.rmtree',side_effect=simulate):
            report=catalog.delete([self.dataset,self.export])
        self.assertEqual(len(report['deleted']),1);self.assertEqual(len(report['failed']),1)
        self.assertTrue(self.export.exists())

    def test_references_and_external_root(self):
        result=self.root/'foundationpose_runs/run_20261102_120200';result.mkdir(parents=True)
        (result/'job.json').write_text(json.dumps(dict(dataset=str(self.dataset),mesh=str(self.base/'model.obj'))))
        external=self.base/'external';external.mkdir();(external/'rgb').mkdir();(external/'cam_K.txt').write_text('K')
        catalog=Catalog(self.root,[external]).scan()
        self.assertEqual(len(catalog.dependents([self.dataset])),2)
        self.assertIn(external,catalog.entries)
        catalog.delete([external]);self.assertFalse(external.exists())


if __name__=='__main__':unittest.main(verbosity=2)
