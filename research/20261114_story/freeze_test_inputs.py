"""Bind all fitted conditions before the once-frozen test evaluation."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent


def main():
    output=ROOT/'TEST_FREEZE.json'
    if output.exists(): raise FileExistsError(output)
    inputs={}
    for dataset in ('MNIST','CIFAR10'):
        for seed in (41,42,43):
            run=ROOT/'runs'/'confirmation'/f'{dataset}{seed}'
            done=json.loads((run/'DONE.json').read_text())
            assert done['status']=='complete' and done['rows']==24 and not done['test_loaded']
            for path in [run/'metadata.json',run/'DONE.json',run/'results.json',*run.glob('*_parameters.npz')]:
                inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('evaluate_frozen_test.py','calibration_core.py','benchmark_cost.py','evaluate_training.py'):
        inputs[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    value=dict(status='frozen_before_new_test_inference',datasets=['MNIST','CIFAR10'],teacher_seeds=[41,42,43],
               T=[1,2,4,8],methods=['baseline','node','graph','mix','output','shuffled'],test_graphs_per_task=10000,
               test_batch_size=128,calibration_samples=512,updates=64,lr=0.001,optimizer='Adam',
               rule='No training, parameter selection, hyperparameter changes or method deletion after this freeze; all endpoints retained',
               qualification='Historically exposed official test, descriptive endpoints, not an untouched holdout',sha256=inputs)
    output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print('Frozen',len(inputs),'input artifacts')


if __name__=='__main__':main()
