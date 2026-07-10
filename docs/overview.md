# Overview

Project này là framework thực nghiệm Federated Learning xây trên Flower và PyTorch. Mục tiêu chính là tạo một codebase dễ đọc, dễ chạy lại thí nghiệm, và dễ mở rộng khi cần thêm baseline mới.

Framework hiện có ba extension point chính:

- `Dataset`: định nghĩa cách load dữ liệu, chia dữ liệu cho client, và tạo `DataLoader`.
- `Model`: định nghĩa cách khởi tạo model PyTorch.
- `Algorithm`: định nghĩa cách tạo Flower server strategy cho từng thuật toán FL.

Các dataset hiện có gồm `mnist`, `fmnist`, `emnist`, `cifar10`, `cifar100`, và `imagenet`.

Các model vision hiện có gồm `mnist_cnn`, `lenet`, `resnet9`, `resnet18`, `resnet34`, và `inception`. Nhóm model mới đọc chung các config `input-channels`, `input-height`, `input-width`, `num-classes` để dễ dùng với dataset khác nhau.

Phiên bản hiện tại có baseline đầu tiên:

| Baseline | Algorithm | Dataset | Model | Partition |
| --- | --- | --- | --- | --- |
| FedAvg MNIST CNN | FedAvg | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedAvgM MNIST CNN | FedAvgM | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedAdagrad MNIST CNN | FedAdagrad | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedAdam MNIST CNN | FedAdam | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedYogi MNIST CNN | FedYogi | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedAdp MNIST CNN | FedAdp | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| GAMF MNIST CNN | GAMF | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedMA MNIST CNN | FedMA | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedGEN MNIST CNN | FedGEN | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedCDA MNIST CNN | FedCDA | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedDRL MNIST CNN | FedDRL | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| Ditto MNIST CNN | Ditto | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedDC MNIST CNN | FedDC | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedDecorr MNIST CNN | FedDecorr | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedEnt MNIST CNN | FedEnt | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedLAW MNIST CNN | FedLAW | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedAAW MNIST CNN | FedAAW | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedDisco MNIST CNN | FedDisco | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedVCK MNIST CNN | FedVCK | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedDyn MNIST CNN | FedDyn | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedExP MNIST CNN | FedExP | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedSAM MNIST CNN | FedSAM | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedSpeed MNIST CNN | FedSpeed | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedNTD MNIST CNN | FedNTD | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedLC MNIST CNN | FedLC | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedRS MNIST CNN | FedRS | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedLAMA MNIST CNN | FedLAMA | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedProto MNIST CNN | FedProto | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedMeta MNIST CNN | FedMeta | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedCurv MNIST CNN | FedCurv | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedMMD MNIST CNN | FedMMD | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedNP MNIST CNN | FedNP | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| APFL MNIST CNN | APFL | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedNova MNIST CNN | FedNova | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| pFedMe MNIST CNN | pFedMe | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedPer MNIST CNN | FedPer | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedRep MNIST CNN | FedRep | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedALA MNIST CNN | FedALA | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedAMP MNIST CNN | FedAMP | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedLAA MNIST CNN | FedLAA | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| FedProx MNIST CNN | FedProx | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| SCAFFOLD MNIST CNN | SCAFFOLD | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |
| MOON MNIST CNN | MOON | MNIST | CNN nhỏ cho ảnh 28x28 | IID, Dirichlet |

Thiết kế cốt lõi: app entrypoints của Flower chỉ đóng vai trò orchestration. Khi thêm baseline mới, ta ưu tiên thêm builder mới và đăng ký vào registry, thay vì sửa logic trong `client_app.py` hoặc `server_app.py`.
