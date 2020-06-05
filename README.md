# Tiresias: Predicting Security Events Through Deep Learning
Pytorch implementation of [Tiresias: Predicting Security Events Through Deep Learning](https://doi.org/10.1145/3243734.3243811) (CCS'18).
This code was implemented as part of the [TODO](TODO) paper. We ask people to [cite](#References) both works when using the software for academic research papers.

## Introduction
With the increased complexity of modern computer attacks, there is a need for defenders not only to detect malicious activity as it happens, but also to predict the specific steps that will be taken by an adversary when performing an attack. However this is still an open research problem, and previous research in predicting malicious events only looked at binary outcomes (eg. whether an attack would happen or not), but not at the specific steps that an attacker would undertake. To fill this gap we present Tiresias xspace, a system that leverages Recurrent Neural Networks (RNNs) to predict future events on a machine, based on previous observations. We test Tiresias xspace on a dataset of 3.4 billion security events collected from a commercial intrusion prevention system, and show that our approach is effective in predicting the next event that will occur on a machine with a precision of up to 0.93. We also show that the models learned by Tiresias xspace are reasonably stable over time, and provide a mechanism that can identify sudden drops in precision and trigger a retraining of the system. Finally, we show that the long-term memory typical of RNNs is key in performing event prediction, rendering simpler methods not up to the task.

## Documentation
We provide an extensive documentation including installation instructions and reference at [tiresias.readthedocs.io](https://tiresias.readthedocs.io/en/latest)

Note, currently the readthedocs is not online.
The docs are available from the `/docs` directory. Build them by executing
```
make build
```
from the `/docs` directory. The documentation can then be found under `/docs/build/html/index.html`. **Important: to build the documentation yourself, you will also need to have `sphinx-rdt-theme` and `recommonmark` installed.**

## References
[1] `TODO`

[2] `Shen, Y., Mariconti, E., Vervier, P. A., & Stringhini, G. (2018). Tiresias: Predicting security events through deep learning. In Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS) (pp. 592-605).`

### Bibtex
```
TODO
```

```
@inproceedings{shen2018tiresias,
  title={Tiresias: Predicting security events through deep learning},
  author={Shen, Yun and Mariconti, Enrico and Vervier, Pierre Antoine and Stringhini, Gianluca},
  booktitle={Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security},
  pages={592--605},
  year={2018}
}
```
