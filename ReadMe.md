# Documentation

## How to use the deployed version
1. Make sure you have installed Anaconda in your PC and XAMPP (or Laragon)
2. You can git clone https://github.com/yehezkielgunawan/My-Skripsi.git
3. Please note this, look at the comment code at **app.py** so you can adapt the pre-trained or the Random Forest model with your own, because I can't deploy it into this repository for the size is too big. You can download the pre-trained model from FastText official website **https://fasttext.cc/docs/en/crawl-vectors.html**
4. Turn on your XAMPP, and turn on your Apache and MYSQL Database.
5. First thing first, make the MYSQL database with the name **sentimen** *(you can adapt it with your own actually)*, make just one table named **data** with one column named **COMMENT** as TEXT *(that is the data type)*
6. After you have cloned this code, you can go into the directories and run this command with your Anaconda Prompt.
```
python app.py
```
7. After that, please wait for a while until **127.0.0.1** *(this can be changed according to your setting in your PC)* appears on your prompt. And you can open the link and enjoy it :D.