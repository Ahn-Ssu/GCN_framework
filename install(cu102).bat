echo "conda update -n base -c defaults conda -y"


conda update -n base -c defaults conda -y


echo "conda update conda --all -y "


conda update conda --all -y 



echo "conda install pytorch torchvision torchaudio cudatoolkit=''MANUAL'' -c pytorch -y"

@REM conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y / /conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge



echo "conda install pandas  -y"

conda install pandas  -y


echo "conda install -c conda-forge tqdm -y"

conda install -c conda-forge tqdm -y


echo "conda install -c conda-forge mendeleev=0.5.2 -y"

conda install -c conda-forge mendeleev=0.5.2 -y


echo "conda install -c anaconda scikit-learn -y"

conda install -c anaconda scikit-learn -y



echo "conda install -c anaconda py-xgboost -y"

conda install -c anaconda py-xgboost -y


@REM https://data.pyg.org/whl/torch-1.12.0+cu116.html 여기만 되고 있음 113+116 존재 x 
@REM pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cu102.html


@REM pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cu102.html


@REM pip install --no-index torch-cluster -f -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cu102.html


@REM pip install --no-index torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cu102.html


@REM pip install torch-geometric



echo "conda install pyg -c pyg -c conda-forge -y"

conda install pyg -c pyg -c conda-forge -y


echo "conda install -c conda-forge rdkit  -y"

conda install -c conda-forge rdkit  -y

echo "conda install -c anaconda xlrd -y"

conda install -c anaconda xlrd -y

echo "conda install -c anaconda openpyxl -y"

conda install -c anaconda openpyxl -y


@REM conda install -c conda-forge deepchem


echo "pip install deepchem"

pip install deepchem

@REM deepchem operating requirement

echo "conda install tensorflow -y"

conda install tensorflow -y

