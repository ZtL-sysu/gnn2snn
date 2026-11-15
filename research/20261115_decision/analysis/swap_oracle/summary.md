# Two-event reassignment capacity diagnostic

training-only optimistic oracle; selection and evaluation use same graphs; descriptive bootstrap is not generalization

|Dataset|Budget|Policy|T|Accuracy %|Teacher agreement %|Feasible graphs|Selected swaps|Total events / baseline|
|---|---|---|---:|---:|---:|---:|---:|---|
|MNIST|exact_events|local|1|78.1250|78.9062|0|0|2130541/2130541|
|MNIST|exact_events|local|2|94.5312|96.0938|0|0|4351056/4351056|
|MNIST|exact_events|message|1|78.1250|78.9062|0|0|2130541/2130541|
|MNIST|exact_events|message|2|94.5312|96.0938|0|0|4351056/4351056|
|MNIST|exact_events|shuffled|1|78.1250|78.9062|0|0|2130541/2130541|
|MNIST|exact_events|shuffled|2|94.5312|96.0938|0|0|4351056/4351056|
|MNIST|exact_events|oracle|1|78.1250|78.9062|0|0|2130541/2130541|
|MNIST|exact_events|oracle|2|94.5312|96.0938|0|0|4351056/4351056|
|MNIST|nonincrease|local|1|78.1250|78.9062|35|0|2130541/2130541|
|MNIST|nonincrease|local|2|94.5312|96.0938|35|0|4351056/4351056|
|MNIST|nonincrease|message|1|78.1250|78.9062|35|9|2130370/2130541|
|MNIST|nonincrease|message|2|94.5312|96.0938|35|9|4350698/4351056|
|MNIST|nonincrease|shuffled|1|78.1250|78.9062|35|10|2130418/2130541|
|MNIST|nonincrease|shuffled|2|94.5312|96.0938|35|10|4350797/4351056|
|MNIST|nonincrease|oracle|1|78.9062|79.6875|35|20|2129828/2130541|
|MNIST|nonincrease|oracle|2|95.3125|96.8750|35|20|4349867/4351056|
|MNIST|first_site_only|local|1|78.1250|78.9062|123|1|2130543/2130541|
|MNIST|first_site_only|local|2|94.5312|96.0938|123|1|4351059/4351056|
|MNIST|first_site_only|message|1|76.5625|77.3438|123|87|2130687/2130541|
|MNIST|first_site_only|message|2|95.3125|96.8750|123|87|4351142/4351056|
|MNIST|first_site_only|shuffled|1|77.3438|78.1250|123|81|2130390/2130541|
|MNIST|first_site_only|shuffled|2|95.3125|96.8750|123|81|4350820/4351056|
|MNIST|first_site_only|oracle|1|82.0312|82.8125|123|77|2131069/2130541|
|MNIST|first_site_only|oracle|2|95.3125|96.8750|123|77|4351959/4351056|
|CIFAR10|exact_events|local|1|54.6875|67.9688|0|0|3407109/3407109|
|CIFAR10|exact_events|local|2|66.4062|85.1562|0|0|6876233/6876233|
|CIFAR10|exact_events|message|1|54.6875|67.9688|0|0|3407109/3407109|
|CIFAR10|exact_events|message|2|66.4062|85.1562|0|0|6876233/6876233|
|CIFAR10|exact_events|shuffled|1|54.6875|67.9688|0|0|3407109/3407109|
|CIFAR10|exact_events|shuffled|2|66.4062|85.1562|0|0|6876233/6876233|
|CIFAR10|exact_events|oracle|1|54.6875|67.9688|0|0|3407109/3407109|
|CIFAR10|exact_events|oracle|2|66.4062|85.1562|0|0|6876233/6876233|
|CIFAR10|nonincrease|local|1|54.6875|67.9688|43|0|3407109/3407109|
|CIFAR10|nonincrease|local|2|66.4062|85.1562|43|0|6876233/6876233|
|CIFAR10|nonincrease|message|1|54.6875|68.7500|43|18|3406821/3407109|
|CIFAR10|nonincrease|message|2|66.4062|85.1562|43|18|6875766/6876233|
|CIFAR10|nonincrease|shuffled|1|53.9062|67.9688|43|19|3406856/3407109|
|CIFAR10|nonincrease|shuffled|2|66.4062|85.1562|43|19|6875734/6876233|
|CIFAR10|nonincrease|oracle|1|54.6875|70.3125|43|17|3406822/3407109|
|CIFAR10|nonincrease|oracle|2|66.4062|85.1562|43|17|6875616/6876233|
|CIFAR10|first_site_only|local|1|54.6875|67.9688|122|0|3407109/3407109|
|CIFAR10|first_site_only|local|2|66.4062|85.1562|122|0|6876233/6876233|
|CIFAR10|first_site_only|message|1|53.9062|68.7500|122|90|3407070/3407109|
|CIFAR10|first_site_only|message|2|67.1875|85.9375|122|90|6876311/6876233|
|CIFAR10|first_site_only|shuffled|1|54.6875|69.5312|122|98|3407193/3407109|
|CIFAR10|first_site_only|shuffled|2|67.1875|85.9375|122|98|6876503/6876233|
|CIFAR10|first_site_only|oracle|1|57.0312|75.7812|122|76|3407524/3407109|
|CIFAR10|first_site_only|oracle|2|66.4062|85.1562|122|76|6877113/6876233|

Capacity continuation gate: **PASS**. This is not a deployment/method gate or independent structure proof. All candidate logits, labels, graph IDs, coordinate pairs and costs are retained.
