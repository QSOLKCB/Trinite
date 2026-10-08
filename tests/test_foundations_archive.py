import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('restore_review',ROOT/'scripts/restore_foundations_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class ArchiveTests(unittest.TestCase):
    def make(self,root,files=None,mutation=None):
        files=files or {'run/summary.json':b'{}','producer/source.py':b'{}'}
        index={name:{'bytes':len(raw),'identity':a.digest(raw)} for name,raw in files.items()}
        members={'objects/sha256/'+a.digest(raw)[7:]:raw for raw in files.values()}
        if mutation:mutation(index,members)
        members['restore-index.json']=json.dumps({'schema':a.SCHEMA,'files':index}).encode()
        path=root/'archive.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for name,raw in members.items():z.writestr(name,raw)
        return path,a.digest(path.read_bytes())

    def test_shared_objects_restore_every_exact_original_file_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive,key=self.make(root);dest=root/'restored'
            result=a.restore(archive,dest,key);self.assertEqual(result['restored_files'],2)
            self.assertEqual((dest/'run/summary.json').read_bytes(),b'{}')
            self.assertEqual((dest/'producer/source.py').read_bytes(),b'{}')
            with self.assertRaises(ValueError):a.restore(archive,dest,key)

    def test_wrong_identity_changed_missing_extra_and_oversized_objects_reject_before_writes(self):
        mutations=(lambda i,m:m.update({next(iter(m)):b'changed'}),lambda i,m:m.clear(),
                   lambda i,m:m.update({'extra':b'extra'}),lambda i,m:i['run/summary.json'].update(bytes=32*1024*1024+1))
        for mutation in mutations:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);archive,key=self.make(root,mutation=mutation);dest=root/'dest'
                with self.assertRaises(ValueError):a.restore(archive,dest,key)
                self.assertFalse(dest.exists())
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive,key=self.make(root)
            with self.assertRaises(ValueError):a.restore(archive,root/'dest','sha256:'+'0'*64)
            self.assertFalse((root/'dest').exists())

    def test_traversal_noncanonical_case_and_parent_collisions_reject(self):
        cases=({'run/../escape':b'x'},{'/run/file':b'x'},{'run/./file':b'x'},{'run\\file':b'x'},
               {'run/A':b'x','run/a':b'x'},{'run/a':b'x','run/a/b':b'x'})
        for files in cases:
            with self.subTest(files=files),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);archive,key=self.make(root,files)
                with self.assertRaises(ValueError):a.restore(archive,root/'dest',key)
                self.assertFalse((root/'dest').exists())

    def test_casefolded_file_ancestors_reject_in_both_index_orders_before_staging(self):
        for ancestor,descendant in (('run/A','run/a/b'),
                                    ('run/Parent/FILE','run/parent/file/deep/child'),
                                    ('run/Straße','run/STRASSE/b')):
            for names in ((ancestor,descendant),(descendant,ancestor)):
                with self.subTest(names=names),tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);archive,key=self.make(root,{name:b'x' for name in names})
                    dest=root/'uncreated-parent'/'dest'
                    with patch.object(a.tempfile,'TemporaryDirectory') as staging:
                        with self.assertRaisesRegex(ValueError,'unsafe/colliding restore path'):
                            a.restore(archive,dest,key)
                        staging.assert_not_called()
                    self.assertFalse(dest.parent.exists())

    def test_casefolded_ancestor_check_keeps_distinct_path_components_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);files={'run/A':b'file','run/ab/b':b'child',
                                      'producer/parent/left':b'left','producer/parent/right':b'right'}
            archive,key=self.make(root,files);dest=root/'restored'
            self.assertEqual(a.restore(archive,dest,key)['restored_files'],len(files))
            for name,raw in files.items():self.assertEqual((dest/name).read_bytes(),raw)


if __name__=='__main__':unittest.main()
