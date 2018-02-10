# How to deploy

1. `jekyll build`
2. `rsync -r _site/ alan@adicu.com:/home/alan/learn.devfe.st`
3. `ssh alan@adicu.com`
4. `sudo rm /srv/learn.devfe.st/public_html`
5. `sudo mv /home/alan/learn.devfe.st/* /srv/learn.devfe.st/public_html/
