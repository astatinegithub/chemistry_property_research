# ADMET prediction Model 
- This project is to create a model that makes predictions based on SMILE data using DNPNN.
- 본 프로젝트는 smiles 데이터셋을 이용하여 admet을 예측해주는 프로젝트 입니다.

> !주의 코드에 아직 주석이 달려있지 않아서 해석에 곤욕을 겪을 수 있음


# 연구 가설
> **H1.** stereochemistry 정보를 atom/bond feature에 포함하면 stereo 정보를 포함하지 않은 모델보다 특히 stereochemical subset에서 ADMET 예측 성능이 향상될 것이다.

> **H2.** stereocenter와 그 주변 local representation을 별도의 attention mechanism으로 집약하면, 단순히 stereo feature만 제공하는 모델보다 stereochemical subset의 ADMET 예측 성능이 향상될 것이다.



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
