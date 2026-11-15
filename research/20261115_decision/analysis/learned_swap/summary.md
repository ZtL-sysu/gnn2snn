# Learned single-pass reassignment: bounded prototype results

teacher41, exposed val512, one fixed ridge/pair family; no test/multi-teacher/external confirmation

|Dataset|Arm|T1 accuracy/agreement %|T2 accuracy/agreement %|T1/T2 actual events|Warm/cold ms per graph (T1)|
|---|---|---:|---:|---|---:|
|MNIST|baseline|78.1250/81.8359|89.4531/93.9453|8585208/17537975|0.691/0.691|
|MNIST|nongraph_local|77.9297/82.0312|89.8438/94.1406|8585138/17538021|4.883/5.057|
|MNIST|graph_local|77.9297/82.0312|90.0391/94.3359|8585181/17538130|4.665/5.369|
|MNIST|nongraph_decision|77.9297/82.0312|89.6484/94.3359|8585111/17537983|4.480/4.809|
|MNIST|graph_decision|78.1250/81.8359|89.8438/94.1406|8585010/17537660|5.050/6.212|
|MNIST|shuffled_decision|78.5156/82.6172|89.0625/94.5312|8584979/17537581|5.015/4.799|
|MNIST|toppin_decision|78.1250/81.8359|89.4531/94.1406|8584939/17537522|4.336/6.026|
|CIFAR10|baseline|49.4141/69.5312|53.9062/84.5703|13547874/27353634|1.463/1.463|
|CIFAR10|nongraph_local|49.2188/69.5312|52.9297/83.3984|13547771/27353513|6.994/7.671|
|CIFAR10|graph_local|49.2188/69.1406|53.1250/83.5938|13547677/27353362|7.207/7.198|
|CIFAR10|nongraph_decision|49.2188/68.5547|53.3203/83.7891|13547783/27353561|7.200/7.529|
|CIFAR10|graph_decision|49.0234/68.7500|52.9297/83.7891|13547962/27353965|7.248/7.796|
|CIFAR10|shuffled_decision|49.0234/68.7500|53.1250/83.9844|13547859/27353754|7.193/7.214|
|CIFAR10|toppin_decision|49.2188/68.7500|52.9297/83.9844|13547917/27353972|7.331/7.136|

Predeclared method gate: **FAIL**. One alignment audit completed without refitting/tuning. Full costs/paired intervals/constraint violations are in CSV and audit JSON. First-site conservation is not a guarantee of downstream costs.
