"""Uncompressed archive policy and mutually exclusive mode controls. No science."""
import copy,os,zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from control_helpers import U,R,HERE
import owner as O
PACKET=Path(os.environ['PUBLIC15_TEST_PACKET']).resolve()


def record_directory(root,size,mode):
 root.mkdir();m=U.read(PACKET/'manifest.json');r={'scope':'SYNTHETIC_ARCHIVE_ONLY_NOT_DEVICE','experiment':mode,**{k:m[k]for k in U.FIELDS}};U.write(root/'receipt.json',r)
 with(root/'synthetic-zero-blob.bin').open('wb')as f:f.truncate(size)
 return r


def test_public_105mib_total_production_pack_and_host_recovery(tmp_path):
 root=tmp_path/'record';r=record_directory(root,105*1024*1024,U.MODE);r.update(R.package(root));inventory=O.verify_result(root,r)
 assert 100*1024*1024<sum(v['bytes']for v in inventory.values())<U.RESULT_MAX_BYTES
 assert(root/'recovered/synthetic-zero-blob.bin').stat().st_size==105*1024*1024
 assert(root/'evidence.zip').stat().st_size<1024*1024

@pytest.mark.parametrize('mode',[R.NC.MODE,R.CC.MODE])
def test_existing_modes_keep_100mib_limit(tmp_path,mode):
 root=tmp_path/'record';record_directory(root,101*1024*1024,mode)
 with pytest.raises(ValueError,match='RESULT_SIZE'):R.package(root)
 assert not(root/'evidence.zip').exists()


def test_public_over_256mib_uncompressed_rejected_before_zip(tmp_path):
 root=tmp_path/'record';record_directory(root,U.RESULT_MAX_BYTES+1,U.MODE)
 with pytest.raises(ValueError,match='PUBLIC_RESULT_TOTAL_UNCOMPRESSED_BOUND'):R.package(root)
 assert not(root/'evidence.zip').exists()

@pytest.mark.parametrize('fault',['total','count','zip_total','schema'])
def test_host_public_result_limit_independent_of_compressed_size(fault):
 inventory={'one':{'sha256':'a'*64,'bytes':1}};infos=[SimpleNamespace(file_size=1)]
 if fault=='total':inventory['one']['bytes']=U.RESULT_MAX_BYTES+1
 if fault=='count':inventory={str(n):{'sha256':'a'*64,'bytes':0}for n in range(U.RESULT_MAX_MEMBERS+1)}
 if fault=='zip_total':infos[0].file_size=U.RESULT_MAX_BYTES+1
 if fault=='schema':inventory['one']['bytes']=-1
 with pytest.raises(ValueError,match='PUBLIC_RESULT_'):U.result_limits(inventory,infos)

@pytest.mark.parametrize('mode',[R.CC.MODE,R.NC.MODE])
def test_public_fields_reject_compiler_and_plain_modes(mode):
 m=U.read(PACKET/'manifest.json');m['experiment']=mode
 with pytest.raises(ValueError,match='PUBLIC_NOT_REQUESTED'):R.verify_experiment(m)

@pytest.mark.parametrize('field',['compiler_generation','compiler_policy_sha256','actual_native_acceptance'])
def test_compiler_only_fields_rejected_by_public_mode(field):
 m=U.read(PACKET/'manifest.json');m[field]='synthetic unwanted field'
 with pytest.raises(ValueError,match='COMPILER_NOT_REQUESTED'):R.verify_experiment(m)
