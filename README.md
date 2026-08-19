# ADMET prediction Model
- 본 프로젝트는 smiles 데이터셋을 이용하여 admet을 예측해주는 프로젝트 입니다.

> !주의 코드에 아직 주석이 달려있지않아서 해석에 곤욕을 겪을수 있음


# 연구 가설

> stereo tag를 가진 원자들을 가지고 stero 보정을 위한 attention을 넣어주면 성능이 향상될 것이다. 



# 나중에 할일
- `dataset_validation.py`이해하기 -> 이해부족으로 활용이 어려움



# dataset
- [PubChem api](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/)
- [chembl](https://www.ebi.ac.uk/chembl/)

# piplist
- setuptools
- torch
- torch_geometric
- torchvision
- pandas
- rdkit
- matplotlib
