#wget --content-disposition https://www.dropbox.com/s/xkijidyu3dhqkwy/11101554.h5?dl=1

wget -O model_1.h5  https://www.dropbox.com/s/xkijidyu3dhqkwy/11101554.h5?dl=1
wget -O model_2.h5  https://www.dropbox.com/s/kqw6f245tr3rrsj/11132310.h5?dl=1
wget -O model_3.h5  https://www.dropbox.com/s/r1xbcmzx0hoq4wn/11150008.h5?dl=1
python3 readfile.py $1
python3 predict.py $2 
