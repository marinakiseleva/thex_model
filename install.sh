cd ..

mkdir libraries
mkdir environments

virtualenv environments/thex_env
source environments/thex_env/bin/activate


cd libraries
git clone https://github.com/marinakiseleva/hmc.git
cd hmc
python setup.py install
cd ../../thex_model

pip3 install -r requirements.txt
python3 setup.py develop
python3 -m ipykernel install --user --name "thexkernel" --display-name "THEx env (py3env)"

cd notebooks
jupyter notebook
