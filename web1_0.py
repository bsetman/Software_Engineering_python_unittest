import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tldextract
import time

# Установка глобальных переменных
visited = set()
broken_links = []
internal_links = set()
external_links = set()
file_links = set()
subdomains = set()
max_pages = 300


def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ['http', 'https']


def normalize_url(url):
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


def crawl(url, base_domain):
    global visited
    if url in visited or len(visited) >= max_pages:
        return
    print(f"[+] Посетить：{url}")
    visited.add(url)

    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            print(f"[x] Неработающие ссылки：{url}")
            broken_links.append(url)
            return
    except Exception as e:
        print(f"[!] Ошибка запроса：{url} - {e}")
        broken_links.append(url)
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup.find_all('a', href=True):
        href = tag['href']
        full_url = urljoin(url, href)
        if not is_valid_url(full_url):
            continue

        ext = tldextract.extract(full_url)
        base_registered = f"{ext.domain}.{ext.suffix}"

        if base_registered == base_domain:
            internal_links.add(full_url)
            if ext.subdomain and ext.subdomain != 'www':
                subdomains.add(ext.subdomain + '.' + base_registered)
            crawl(full_url, base_domain)
        else:
            external_links.add(full_url)

        # Проверьте, является ли это ссылкой на файл
        if any(full_url.endswith(ext) for ext in ['.pdf', '.doc', '.docx']):
            file_links.add(full_url)


def get_user_input():
    while True:
        url = input("Введите URL-адрес веб-сайта для сканирования (например: spbu.ru или https://spbu.ru): ").strip()
        if not url:
            print("Ошибка: URL не может быть пустым")
            continue

        # Канонизировать URL-адреса
        normalized_url = normalize_url(url)

        try:
            parsed = urlparse(normalized_url)
            if not parsed.netloc:
                print("Ошибка: неверный формат URL")
                continue

            ext = tldextract.extract(normalized_url)
            if not ext.domain or not ext.suffix:
                print("Ошибка: Невозможно получить основной домен")
                continue

            base_domain = f"{ext.domain}.{ext.suffix}"
            return normalized_url, base_domain
        except Exception as e:
            print(f"Ошибка анализа URL: {e}")
            continue


# Главный вход
if __name__ == '__main__':
    start_time = time.time()

    print("===Инструменты для сканирования веб-сайтов===")
    print(f"Примечание: будет просканировано не более {max_pages} страниц.")
    start_url, domain = get_user_input()

    print(f"\nНачать сканирование: {start_url} (Основное доменное имя: {domain})")
    crawl(start_url, domain)

    # Вывод статистических результатов
    print("\n --- Статистические результаты ---")
    print(f"Общее количество посещений страницы: {len(visited)}")
    print(f"Внутренние страницы: {len(internal_links)}")
    print(f"Количество недействительных страниц: {len(broken_links)}")
    print(f"Количество поддоменов: {len(subdomains)} ({subdomains})")
    print(f"Количество внешних ссылок: {len(external_links)}")
    print(f"Уникальные ссылки на документы: {len(file_links)}")
    print(f"Время：{round(time.time() - start_time, 2)} секунд")