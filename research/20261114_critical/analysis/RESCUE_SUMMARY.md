# Source baseline and allocation rescue

All values: exploratory validation512, teacher41, no test. T1-fit base offsets are shared across T1/T2.

|Dataset|T|Policy|Accuracy%|Extra nonzero events|Opportunity slots|
|---|---:|---|---:|---:|---:|
|CIFAR10|1|percentile_v2/author_cfg|22.4609|0|0|
|CIFAR10|2|percentile_v2/author_cfg|32.8125|0|0|
|CIFAR10|1|percentile_v2/matched8|46.2891|0|0|
|CIFAR10|2|percentile_v2/matched8|55.0781|0|0|
|MNIST|1|percentile_v2/author_cfg|23.4375|0|0|
|MNIST|2|percentile_v2/author_cfg|49.2188|0|0|
|MNIST|1|percentile_v2/matched8|71.8750|0|0|
|MNIST|2|percentile_v2/matched8|88.0859|0|0|
|CIFAR10|1|allocation/baseline|47.8516|0|0|
|CIFAR10|2|allocation/baseline|53.9062|0|0|
|CIFAR10|1|allocation/random|48.2422|814867|2225186|
|CIFAR10|2|allocation/random|54.8828|814867|2225186|
|CIFAR10|1|allocation/local|48.6328|926683|2225186|
|CIFAR10|2|allocation/local|52.3438|926683|2225186|
|CIFAR10|1|allocation/identity|49.2188|923163|2225186|
|CIFAR10|2|allocation/identity|53.1250|923163|2225186|
|CIFAR10|1|allocation/shuffled|47.2656|924819|2225186|
|CIFAR10|2|allocation/shuffled|52.7344|924819|2225186|
|CIFAR10|1|allocation/message|49.2188|926039|2225186|
|CIFAR10|2|allocation/message|53.3203|926039|2225186|
|MNIST|1|allocation/baseline|75.0000|0|0|
|MNIST|2|allocation/baseline|89.2578|0|0|
|MNIST|1|allocation/random|76.3672|562585|1344222|
|MNIST|2|allocation/random|89.6484|562585|1344222|
|MNIST|1|allocation/local|77.1484|610926|1344222|
|MNIST|2|allocation/local|89.6484|610926|1344222|
|MNIST|1|allocation/identity|76.7578|608911|1344222|
|MNIST|2|allocation/identity|89.4531|608911|1344222|
|MNIST|1|allocation/shuffled|79.2969|609872|1344222|
|MNIST|2|allocation/shuffled|89.4531|609872|1344222|
|MNIST|1|allocation/message|79.4922|612232|1344222|
|MNIST|2|allocation/message|88.6719|612232|1344222|

Predefined expansion gate: FAIL. No stable method claim.
