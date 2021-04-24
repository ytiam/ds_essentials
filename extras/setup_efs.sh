sudo apt install nfs-kernel-server
sudo exportfs -a
mkdir data
cd data
sudo chmod go+rw .
cd
