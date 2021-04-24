cd
mkdir GIT
cd GIT
git clone https://github.com/toobaz/ipynb_output_filter.git
chmod +x ~/GIT/ipynb_output_filter/ipynb_output_filter.py
touch ~/.gitattributes
echo "*.ipynb    filter=dropoutput_ipynb" >> ~/.gitattributes
git config --global core.attributesfile ~/.gitattributes
