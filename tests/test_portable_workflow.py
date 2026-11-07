"""Portable config and actual PNG dataset contract tests, with synthetic inputs only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from data_ops import validate
from segment_plan import ranges, signature, cached_result, cache_file, write_json

class PortableWorkflow(unittest.TestCase):
    def test_relative_config_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config = folder/'settings.json'
            config.write_text(json.dumps({'sam_project':'source','sam_checkpoint':'weights.pt'}))
            env = os.environ.copy()
            # Isolate this test from developer-local configuration.
            env = {k:v for k,v in env.items() if not k.startswith('STUDIO_')}
            env.update(STUDIO_CONFIG=str(config), STUDIO_SAM_CHECKPOINT=str(folder/'override.pt'))
            result = subprocess.check_output([sys.executable, '-c',
                'import config,json; print(json.dumps([str(config.SAM_PROJECT),str(config.SAM_CHECKPOINT)]))'],
                cwd=ROOT, env=env, text=True)
            self.assertEqual(json.loads(result), [str(folder/'source'),str(folder/'override.pt')])

    def test_export_contract_and_reject_incorrect_depth(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);source=folder/'input'
            for sub in ['rgb','depth','masks']:(source/sub).mkdir(parents=True)
            (source/'cam_K.txt').write_text('50 0 32\n0 50 24\n0 0 1\n')
            for i in range(6):
                name=f'{i:06d}.png'
                cv2.imwrite(str(source/'rgb'/name),np.full((48,64,3),100,np.uint8))
                cv2.imwrite(str(source/'depth'/name),np.full((48,64),500,np.uint16))
                cv2.imwrite(str(source/'masks'/name),np.full((48,64),255,np.uint8))
            report=validate(source,export_to=folder/'export')
            self.assertEqual(report['usable'],6)
            export=folder/'export/segment_001'
            self.assertEqual(len(list((export/'rgb').glob('*.png'))),6)
            self.assertEqual(cv2.imread(str(export/'depth/000000.png'),-1).dtype,np.uint16)
            # Unlink first: export may share the inode; never mutate an exported hardlink.
            (source/'depth/000000.png').unlink()
            cv2.imwrite(str(source/'depth/000000.png'),np.full((48,64),50,np.uint8))
            self.assertEqual(validate(source)['usable'],5)
            self.assertEqual(cv2.imread(str(export/'depth/000000.png'),-1).dtype,np.uint16)

    def test_segment_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);files=[]
            for i in range(4):
                p=folder/f'{i}.png';p.write_bytes(b'image');files.append(p)
            plan={'starts':[0,2]}
            self.assertEqual(ranges(files,plan),[(0,2),(2,4)])
            (folder/'cached').mkdir()
            for p in files[:2]:(folder/'cached'/p.name).write_bytes(b'mask')
            prompts={'0.png':[[10,10,1]]}
            write_json(cache_file(folder,files,0),dict(signature=signature(files,prompts,0,2),masks='cached'))
            self.assertIsNotNone(cached_result(folder,files,prompts,0,2))
            prompts['0.png'].append([20,20,0])
            self.assertIsNone(cached_result(folder,files,prompts,0,2))
            with self.assertRaises(ValueError):ranges(files,{'starts':[1]})

if __name__=='__main__':unittest.main()
