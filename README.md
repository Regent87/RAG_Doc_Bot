# RAG_Doc_Bot
Телеграмм бот для поиска по документам

## Функционал
- Поиск по технической документации
- Документы нужна загрузить в папку /documents
- Тестировал только на докумкнтах формата pdf
- Используется модель meta-llama-3-8b-instruct
- модель развернута на сервисе https://replicate.com/
- Протестировать можно тут https://t.me/RogDocBot 




## Установка
```bash
git clone https://github.com/Regent87/RAG_Doc_Botr.git
cd dantex-ac-helper
pip install -r requirements.txt
в main.py установите свои токины:
REPLICATE_API_TOKEN и BOT_TOKEN
BOT_TOKEN для Tg бота
REPLICATE_API_TOKEN для https://replicate.com/





