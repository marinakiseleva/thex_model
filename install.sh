mkdir libraries
cd libraries
git clone https://github.com/marinakiseleva/hmc.git
cd ..
git clone https://github.com/marinakiseleva/thex_model.git
mkdir environments
virtualenv environments/thex_env
source environments/thex_env/bin/activate
cd libraries/hmc
python setup.py install
cd ../../thex_model
pip install -r requirements.txt
python setup.py develop
python -m ipykernel install --user --name thexenv --display-name "THEx env (py3env)"
