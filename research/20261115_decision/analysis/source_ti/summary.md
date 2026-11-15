# Source threshold iteration: raw-derived results

Teacher41, exposed val512 per dataset; source moments/TI/fallback run unchanged. Eight levels and GCN routing are explicit adaptations. H is calibration horizon; one fitted vector is shared at all evaluated T.

|Dataset|H|T|Accuracy %|Teacher agreement %|Actual events|Δ accuracy vs strong local pp [graph95%CI]|NaN fallback channels|
|---|---:|---:|---:|---:|---:|---|---:|
|MNIST|1|1|71.4844|74.2188|8745041|-6.6406 [-10.9375,-2.3438]|35|
|MNIST|1|2|89.6484|97.6562|17724743|+0.1953 [-1.9531,2.3438]|35|
|MNIST|1|4|91.2109|100.0000|30131859|+0.7812 [-0.5859,2.1484]|35|
|MNIST|1|8|91.2109|100.0000|37111627|+0.1953 [-0.7812,1.1719]|35|
|MNIST|2|1|72.0703|75.0000|8746202|-6.0547 [-10.5469,-1.7578]|35|
|MNIST|2|2|89.4531|97.0703|17726861|+0.0000 [-2.3438,2.1484]|35|
|MNIST|2|4|91.0156|99.8047|30132912|+0.5859 [-0.7812,1.9531]|35|
|MNIST|2|8|91.2109|99.8047|37099802|+0.1953 [-0.7812,1.1719]|35|
|CIFAR10|1|1|45.3125|66.4062|13353686|-4.1016 [-8.5938,0.1953]|20|
|CIFAR10|1|2|52.5391|90.6250|26947396|-1.3672 [-4.2969,1.5625]|20|
|CIFAR10|1|4|54.8828|99.6094|45688314|+0.9766 [-0.9766,2.7344]|20|
|CIFAR10|1|8|55.2734|100.0000|54046913|+0.9766 [-0.3906,2.3438]|20|
|CIFAR10|2|1|45.5078|65.4297|13353564|-3.9062 [-8.2031,0.5859]|19|
|CIFAR10|2|2|52.1484|90.4297|26946265|-1.7578 [-4.6875,0.9766]|19|
|CIFAR10|2|4|54.6875|99.4141|45683326|+0.7812 [-0.9766,2.7344]|19|
|CIFAR10|2|8|55.2734|100.0000|54051475|+0.9766 [-0.3906,2.3438]|19|

These are source-algorithm controls, not a new graph encoding contribution, a literal ImageNet reproduction, or independent multi-teacher confirmation. No official test dataset or teacher retraining. No source-scale or level search.
