# SetUp GitLab For REHL6
sudo yum install -y curl policycoreutils-python openssh-server cronie
sudo yum install -y lokkit
sudo lokkit -s http -s ssh
sudo yum install postfix
sudo service postfix start
sudo chkconfig postfix on
curl https://packages.gitlab.com/install/repositories/gitlab/gitlab-ee/script.rpm.sh | sudo bash
read -p "EXTERNAL_URL : "
sudo EXTERNAL_URL="http://$REPLY" yum -y install gitlab-ee

