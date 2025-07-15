import os
import re
import time

import numpy as np
import sqlite3
import replicate
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import faiss
import hashlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
from test_questions import questions1 as questions

# Конфигурация
DOCUMENTS_DIR = "documents"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large" #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REPLICATE_API_TOKEN = "Ваш API ключь для replicatr.com"
INDEX_FILE = "faiss_index.index"
DB_FILE = "documents.db"
CHUNK_SIZE = 300
OVERLAP = 120
MAX_DOCUMENTS = 1000
BATCH_SIZE = 32

# Telegram бот
BOT_TOKEN = "Ваш API ключь для tg бота "


class SynonymExpander:
    def __init__(
            self,
            synonym_dict: dict,
            model_name: str = 'intfloat/multilingual-e5-large',
            use_llama: bool = False,
            llama_model: str = "meta/meta-llama-3-8b-instruct",
            replicate_api_token: Optional[str] = None
    ):
        self.model = SentenceTransformer(model_name)
        self.synonym_dict = synonym_dict
        self.use_llama = use_llama
        self.llama_model = llama_model
        self.replicate_api_token = replicate_api_token

        if self.use_llama and not self.replicate_api_token:
            raise ValueError("Replicate API token is required when use_llama=True")

        self._prepare_embeddings()

    def _prepare_embeddings(self) -> None:
        """Кэшируем эмбеддинги всех ключей словаря"""
        self.keys = list(self.synonym_dict.keys())
        self.key_embeddings = self.model.encode(self.keys)

    def find_closest_key(self, term: str, threshold: float = 0.86) -> Optional[str]:
        """Находит ближайший ключ в словаре по косинусной схожести эмбеддингов"""
        term_embedding = self.model.encode([term])
        similarities = cosine_similarity(term_embedding, self.key_embeddings)[0]
        max_idx = np.argmax(similarities)
        return self.keys[max_idx] if similarities[max_idx] > threshold else None

    def _expand_with_llama(self, query: str) -> str:
        """Расширяет запрос через Llama 3, используя Replicate API"""
        try:
            client = replicate.Client(api_token=self.replicate_api_token)

            prompt = f"""
            Расширь следующий технический запрос, добавив синонимы и варианты формулировок на Русском языке.
            Запрос: "{query}"
            Расширенный запрос должен сохранять исходный смысл, но включать альтернативные формулировки ключевых терминов.
            Верни только расширенный запрос без пояснений.
            Пример: "Как включить продув?" → "Как активировать (запустить/включить) режим продува (очистки)?"
            Расширенный запрос:
            """

            output = client.run(
                self.llama_model,
                input={
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_new_tokens": 100
                }
            )
            return "".join(output).strip('"')

        except Exception as e:
            print(f"Ошибка Replicate API: {e}")
            return query  # Возвращаем оригинальный запрос в случае ошибки

    def expand_query(self, query: str) -> str:
        """
        Расширяет запрос двумя способами (на выбор):
        1. Через заранее подготовленный словарь синонимов
        2. Через LLM (Llama 3), если use_llama=True
        """


        # Оригинальный метод расширения через словарь
        expanded_terms = []
        for term in query.split():
            closest_key = self.find_closest_key(term)
            if closest_key:
                synonyms = self.synonym_dict[closest_key]
                expanded_terms.append(term)
                expanded_terms.append(f"({' OR '.join(synonyms)})")
            else:
                expanded_terms.append(term)
        if self.use_llama:
            return self._expand_with_llama(' '.join(expanded_terms))
        else:
            return ' '.join(expanded_terms)

class DocumentDatabase:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.conn = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        tech_synonyms = {
            "испарител": ["x-fan", "режим просушки", "функция продува"],
            #"включить": ["нажать на кнопку"]
            #"rk-24svg": ["dantex 24", "модель 24svg", "серия svgi"],
            #"кнопка": ["клавиша", "переключатель", "сенсор", "кнопочный элемент"]
        }
        self.expander = SynonymExpander(tech_synonyms,use_llama=True, replicate_api_token=REPLICATE_API_TOKEN)

    def initialize_db(self):
        """Инициализация базы данных SQLite"""
        self.conn = sqlite3.connect(DB_FILE)
        cursor = self.conn.cursor()

        # Таблица для документов
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            filehash TEXT NOT NULL,
            processed BOOLEAN DEFAULT 0,
            UNIQUE(filehash)
        )""")

        # Таблица для чанков
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_hash TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY(doc_id) REFERENCES documents(id),
            UNIQUE(chunk_hash)
        )""")

        self.conn.commit()

    def scan_documents(self):
        """Сканирование папки с документами и добавление в БД"""
        cursor = self.conn.cursor()

        for filename in os.listdir(DOCUMENTS_DIR):
            if filename.endswith(".pdf"):
                filepath = os.path.join(DOCUMENTS_DIR, filename)

                # Вычисляем хеш файла для проверки изменений
                with open(filepath, 'rb') as f:
                    filehash = hashlib.md5(f.read()).hexdigest()

                # Проверяем, есть ли уже этот файл в БД
                cursor.execute(
                    "SELECT id FROM documents WHERE filehash = ?",
                    (filehash,)
                )
                if cursor.fetchone() is None:
                    # Добавляем новый документ
                    cursor.execute(
                        "INSERT INTO documents (filename, filepath, filehash) VALUES (?, ?, ?)",
                        (filename, filepath, filehash)
                    )
                    self.conn.commit()

        # Получаем список необработанных документов
        cursor.execute(
            "SELECT id, filepath FROM documents WHERE processed = 0 LIMIT ?",
            (MAX_DOCUMENTS,)
        )
        return cursor.fetchall()

    def process_documents(self):
        """Обработка документов и создание чанков"""
        unprocessed_docs = self.scan_documents()

        for doc_id, filepath in tqdm(unprocessed_docs, desc="Processing documents"):
            try:
                text = extract_text(filepath)
                #text = re.sub(r'\s+', ' ', text).strip()
                #print(text)

                # Разбиваем текст на чанки
                chunks = self._split_into_chunks(text)

                # Сохраняем чанки в БД
                cursor = self.conn.cursor()
                for chunk in chunks:
                    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()

                    # Проверяем, есть ли уже этот чанк
                    cursor.execute(
                        "SELECT id FROM chunks WHERE chunk_hash = ?",
                        (chunk_hash,)
                    )
                    if cursor.fetchone() is None:
                        cursor.execute(
                            "INSERT INTO chunks (doc_id, chunk_text, chunk_hash) VALUES (?, ?, ?)",
                            (doc_id, chunk, chunk_hash)
                        )

                # Помечаем документ как обработанный
                cursor.execute(
                    "UPDATE documents SET processed = 1 WHERE id = ?",
                    (doc_id,)
                )
                self.conn.commit()

            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                continue

    def _split_into_chunks(self, text):
        """Улучшенное разбиение текста на чанки без зависимости от заголовков"""
        # Нормализуем пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text).strip()

        # Разбиваем на предложения (грубая сегментация)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Если предложение слишком длинное, разбиваем его
            if sentence_length > CHUNK_SIZE:
                words = sentence.split()
                for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
                    sub_chunk = ' '.join(words[i:i + CHUNK_SIZE])
                    chunks.append(sub_chunk)
                continue

            # Добавляем предложение к текущему чанку
            if current_length + sentence_length <= CHUNK_SIZE:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Сохраняем текущий чанк и начинаем новый с перекрытием
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-int(OVERLAP / 20):]  # Сохраняем часть для контекста
                current_chunk.append(sentence)
                current_length = sum(len(s) for s in current_chunk)

        # Добавляем последний чанк
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def generate_embeddings(self):
        """Генерация эмбеддингов для всех чанков"""
        cursor = self.conn.cursor()

        # Получаем чанки без эмбеддингов
        cursor.execute(
            "SELECT id, chunk_text FROM chunks WHERE embedding IS NULL"
        )
        chunks = cursor.fetchall()

        # Обрабатываем батчами для экономии памяти
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Generating embeddings"):
            batch = chunks[i:i + BATCH_SIZE]
            chunk_ids = [c[0] for c in batch]
            chunk_texts = [c[1] for c in batch]

            # Генерируем эмбеддинги
            embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Сохраняем в БД
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                cursor.execute(
                    "UPDATE chunks SET embedding = ? WHERE id = ?",
                    (embedding.tobytes(), chunk_id)
                )
            self.conn.commit()

    def build_index(self):
        """Построение FAISS индекса"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT embedding FROM chunks WHERE embedding IS NOT NULL")
        embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]

        if not embeddings:
            raise ValueError("No embeddings found in database")

        embeddings = np.vstack(embeddings)
        dimension = embeddings.shape[1]

        # Используем IndexIVFFlat для больших наборов данных
        quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, min(100, len(embeddings) // 2))

        # Обучаем индекс на части данных
        train_embeddings = embeddings[:min(10000, len(embeddings))]
        self.index.train(train_embeddings)

        # Добавляем все эмбеддинги
        self.index.add(embeddings)
        faiss.write_index(self.index, INDEX_FILE)

    def load_index(self):
        """Загрузка индекса"""
        if os.path.exists(INDEX_FILE):
            self.index = faiss.read_index(INDEX_FILE)
        else:
            self.build_index()

    def enhance_query(self, query):
        """Автоматически добавляет технические уточнения к запросу"""
        tech_terms = {
            'продув': ['X-FAN', 'кнопка продува', 'функция просушки']}

        enhanced = []
        for word in query.split():
            if word.lower() in tech_terms:
                enhanced.append(' OR '.join(tech_terms[word.lower()] + [word]))
            else:
                enhanced.append(word)
        for word in tech_terms:
            if word in query.lower():
                enhanced.append(' OR '.join(tech_terms[word]))
        enhanced.append(query)
        return ' '.join(enhanced)

    def search(self, query, k=10):
        """Поиск релевантных чанков"""
        #query = self.enhance_query(query)

        query = self.expander.expand_query(query)
        print(query)
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        distances, indices = self.index.search(query_embedding, k)
        #print(distances, indices)

        cursor = self.conn.cursor()
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            cursor.execute("""
                SELECT c.chunk_text, d.filename 
                FROM chunks c 
                JOIN documents d ON c.doc_id = d.id 
                WHERE c.id = ?
            """, (int(idx) + 1,))  # SQLite uses 1-based indexing

            row = cursor.fetchone()
            if row:
                chunk_text, filename = row
                results.append({
                    'document': filename,
                    'content': chunk_text,
                    'score': float(distance)
                })
        return results

    def print_chunks(self, doc_name=None, max_chunks=200):
        """Печать чанков для проверки"""
        cursor = self.conn.cursor()

        if doc_name:
            cursor.execute("""
                SELECT c.id, c.chunk_text 
                FROM chunks c 
                JOIN documents d ON c.doc_id = d.id 
                WHERE d.filename = ?
                LIMIT ?
            """, (doc_name, max_chunks))
        else:
            cursor.execute("SELECT id, chunk_text FROM chunks LIMIT ?", (max_chunks,))

        for chunk_id, chunk_text in cursor.fetchall():
            print(f"\n--- Чанк ID {chunk_id} ({len(chunk_text)} символов) ---")
            print(chunk_text[:] + "..." if len(chunk_text) > 200 else chunk_text)

    def test_Rag(self, test_sp, generator):
        for tema in test_sp:
            print(tema)
            for question in test_sp[tema]:
                results = self.search(question)
                if not results:
                    print("Извините, я не нашел информации по вашему вопросу.")
                else:
                    answer = generator.generate_answer(question, results)
                    doc_names = set(res['document'] for res in results)
                    if doc_names:
                        answer += f"\n\nИсточники: {', '.join(doc_names)}"
                    print(answer)
                time.sleep(0.01)


class ReplicateAnswerGenerator:
    def __init__(self):
        self.client = replicate.Client(api_token=REPLICATE_API_TOKEN)

    def generate_answer(self, question, context):
        """Генерация ответа с использованием Llama 2 7B через Replicate"""
        context_str = "\n\n".join([f"Документ: {res['document']}\n{res['content']}" for res in context])

        prompt = f"""Ты - помощник по кондиционерам. Ответь на вопрос на Русском языке, используя предоставленную информацию. 
        При необходимости представь ответ в виде пошаговой инструкции. Напиши все необходимые примечания.
Если ответа нет в информации, скажи, что не знаешь.

Вопрос: {question}

Информация: {context_str}

Ответ:"""

        try:
            output = self.client.run(
                "meta/meta-llama-3-8b-instruct",
                input={
                    "prompt": prompt,
                    "system_prompt": "Ты - эксперт по кондиционерам. Давай точные и краткие ответы на основе предоставленной информации.",
                    "max_new_tokens": 8000,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            )

            # Объединяем потоковый вывод в один текст
            answer = "".join([item for item in output])
            return answer.strip()

        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return "Извините, произошла ошибка при генерации ответа."


TEST = False
# Инициализация компонентов
db = DocumentDatabase()
db.initialize_db()
db.process_documents()
db.generate_embeddings()
db.load_index()
generator = ReplicateAnswerGenerator()
if TEST:
    db.test_Rag(questions, generator)
    exit()

#db.print_chunks("dantex_vita.pdf")



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот-помощник по кондиционерам. Задайте мне вопрос о кондиционерах, и я постараюсь помочь."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text

    # Поиск релевантных фрагментов
    results = db.search(question)
    if not results:
        await update.message.reply_text("Извините, я не нашел информации по вашему вопросу.")
        return

    # Генерация ответа
    answer = generator.generate_answer(question, results)

    # Добавляем информацию об источниках
    doc_names = set(res['document'] for res in results)
    if doc_names:
        answer += f"\n\nИсточники: {', '.join(doc_names)}"

    await update.message.reply_text(answer)


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
