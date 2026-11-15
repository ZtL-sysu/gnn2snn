# Raw-array summary

Teacher41, exposed validation512; identical parameter vector at T1/2/4/8. All objectives are existing controls.

|Dataset|Objective|T1 acc/agreement %|T2 acc/agreement %|T4 acc %|T8 acc %|T1 events|
|---|---|---:|---:|---:|---:|---:|
|MNIST|local_t1|78.1250/81.8359|89.4531/93.9453|90.4297|91.0156|8585208|
|MNIST|local_joint|75.1953/78.5156|88.8672/95.3125|90.6250|91.0156|8659536|
|MNIST|logits|79.2969/82.2266|89.6484/93.9453|90.2344|90.6250|8408606|
|MNIST|kl|78.5156/81.4453|87.5000/92.1875|89.8438|90.8203|8553600|
|MNIST|ranking|53.7109/54.2969|68.7500/71.0938|83.7891|88.2812|9318817|
|CIFAR10|local_t1|49.4141/69.5312|53.9062/84.5703|53.9062|54.2969|13547874|
|CIFAR10|local_joint|48.6328/67.1875|55.6641/90.6250|54.6875|54.6875|13508684|
|CIFAR10|logits|50.5859/67.3828|53.7109/86.9141|54.8828|55.8594|13109364|
|CIFAR10|kl|47.2656/70.1172|51.9531/85.9375|54.6875|55.2734|13032401|
|CIFAR10|ranking|36.7188/43.7500|45.1172/59.9609|51.7578|53.3203|13377426|

Gate2 admission: **FAIL**. See paired.csv for graph bootstrap intervals. No graph-specific new method established.
