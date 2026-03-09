#/bin/bash
echo "Make sure to run as root!"

# The swap file isn't persistent
sudo chmod 600 /swapfile
sudo mkswwawp /swapfile
sudo swapon /swapfile

sudo swapon --show
echo "\n If you don't see an 8Gb swapfile, there's an error"
