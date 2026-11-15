# Bounded temporal selector results

Teacher41, exposed validation512. Both prefixes share final 584 new coefficients. First-site T2 decode is exact; full-network output and costs need not match. T1 error array is a placeholder, not an equality assertion.

|Dataset|Arm|T1 accuracy/agreement %|T2 accuracy/agreement %|T1/T2 events|T1 warm/cold ms per graph|
|---|---|---:|---:|---|---:|
|MNIST|baseline|78.1250/81.8359|89.4531/93.9453|8585208/17537975|0.400/0.400|
|MNIST|nongraph_local|69.1406/73.2422|89.0625/93.5547|8626326/17683137|0.715/0.718|
|MNIST|graph_local|75.5859/77.1484|90.0391/94.9219|8604973/17608348|0.711/0.862|
|MNIST|nongraph_decision|73.0469/77.7344|89.8438/93.9453|8552718/17551294|0.719/0.711|
|MNIST|graph_decision|72.8516/77.3438|88.6719/92.7734|8605620/17596988|0.709/0.862|
|MNIST|shuffled_decision|76.9531/80.2734|89.8438/92.9688|8581249/17551966|0.714/0.898|
|MNIST|toppin_decision|75.5859/79.6875|90.0391/94.9219|8594255/17584612|0.711/0.781|
|CIFAR10|baseline|49.4141/69.5312|53.9062/84.5703|13547874/27353634|0.523/0.523|
|CIFAR10|nongraph_local|47.6562/67.1875|52.9297/86.1328|13605531/27562635|0.948/0.951|
|CIFAR10|graph_local|51.7578/69.7266|52.7344/87.1094|13515861/27355542|0.943/1.132|
|CIFAR10|nongraph_decision|48.6328/70.3125|52.1484/86.5234|13618425/27558477|0.948/0.950|
|CIFAR10|graph_decision|50.7812/70.7031|53.1250/84.7656|13612920/27479950|0.944/1.132|
|CIFAR10|shuffled_decision|49.4141/68.1641|52.3438/84.3750|13637028/27503941|0.937/1.173|
|CIFAR10|toppin_decision|48.0469/70.3125|52.1484/84.5703|13612415/27481232|0.948/1.025|

Gate: FAIL. No post-validation fitting. Full costs, paired graph bootstrap uncertainty, and training alignment are in adjacent CSV/JSON. GPU wall time is not chip energy.
