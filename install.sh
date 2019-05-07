cd ..

mkdir libraries
mkdir environments

virtualenv -p /usr/bin/python3 environments/thex_env
source environments/thex_env/bin/activate


cd libraries
git clone https://github.com/marinakiseleva/hmc.git
cd hmc
python setup.py install
cd ../../thex_model

pip install -r requirements.txt
python setup.py develop
python -m ipykernel install --user --name thexenv --display-name "THEx env (py3env)"

cd notebooks
jupyter notebook
