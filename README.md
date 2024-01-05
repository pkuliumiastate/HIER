# Distributed and Privacy-Preserving Mitigation of Poisoning Attacks in Three-Tier Federated Learning Systems

Code for Paper "Distributed and Privacy-Preserving Mitigation of Poisoning
Attacks in Three-Tier Federated Learning Systems"


---


# Quick Start: 
##To run label flipping attack:
```
python Hier-Local-QSGD.py --dataset cifar10 --num_communication 100 --model cnn_complex --num_clients 150 --num_edges 2  --g 2048 --w 2048 --num_reference -35 --num_honest_client 143 --attack target_attack --alpha 0.05
```
##To run backdoor attack:
```
python Hier-Local-QSGD.py --dataset cifar10 --num_communication 100 --model cnn_complex --num_clients 150 --num_edges 2  --g 2048 --w 2048 --num_reference -35 --num_honest_client 135 --attack backdoor_attack --alpha 1
```

---


## Acknowledgements
As this code is reproduced based on the open-sourced code [Hierarchical Federated Learning With Quantization: Convergence Analysis and System Design](https://github.com/LuminLiu/Hier_QSGD) and [Client-Edge-Cloud Hierarchical Federated Learning](https://github.com/LuminLiu/HierFL), the authors would like to thank their contribution. 
