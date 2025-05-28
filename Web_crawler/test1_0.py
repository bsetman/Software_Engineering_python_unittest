import unittest
from unittest.mock import patch, MagicMock
from web1_0 import (
    normalize_url, is_valid_url, crawl,
    visited, broken_links, internal_links,
    external_links, file_links, subdomains,
)



class TestCrawlerFunctions(unittest.TestCase):
    """Тестирование основных функций"""

    def test_normalize_url(self):
        self.assertEqual(normalize_url("google.com"), "https://google.com")
        self.assertEqual(normalize_url("http://github.com"), "http://github.com")
        self.assertEqual(normalize_url("docs.github.com"), "https://docs.github.com")

    def test_is_valid_url(self):
        self.assertTrue(is_valid_url("https://google.com/search"))
        self.assertTrue(is_valid_url("http://github.com/about"))
        self.assertFalse(is_valid_url("ftp://google.com"))
        self.assertFalse(is_valid_url("javascript:void(0)"))


class BaseCrawlerTest(unittest.TestCase):
    """Тестовый базовый класс, содержащий общие методы"""

    def setUp(self):
        # Сбросить глобальное состояние
        visited.clear()
        broken_links.clear()
        internal_links.clear()
        external_links.clear()
        file_links.clear()
        subdomains.clear()

    def create_mock_response(self, html, status=200):
        mock = MagicMock()
        mock.status_code = status
        mock.text = html
        return mock


class TestGoogleCrawl(BaseCrawlerTest):
    """Тестирование сканирования сайта Google"""

    @patch('requests.get')
    def test_google_crawl(self, mock_get):
        """Тестовое сканирование главной страницы Google"""
        mock_get.return_value = self.create_mock_response(
            """
            <html>
                <a href="/search?q=python">Python Search</a>
                <a href="https://maps.google.com">map</a>
                <a href="https://www.youtube.com">External links</a>
                <a href="/privacy.pdf">Privacy Policy</a>
            </html>
            """
        )

        crawl("https://www.google.com", "google.com")

        self.assertIn("https://www.google.com", visited)
        self.assertIn("https://www.google.com/search?q=python", internal_links)
        self.assertIn("https://maps.google.com", internal_links)
        self.assertIn("https://www.youtube.com", external_links)
        self.assertIn("https://www.google.com/privacy.pdf", file_links)
        self.assertIn("maps.google.com", subdomains)


class TestGitHubCrawl(BaseCrawlerTest):
    """Тестирование сканирования веб-сайта GitHub"""

    @patch('requests.get')
    def test_github_crawl(self, mock_get):
        """Тестовое сканирование домашней страницы GitHub"""
        mock_get.return_value = self.create_mock_response(
            """
            <html>
                <a href="/features">Function</a>
                <a href="https://docs.github.com">document</a>
                <a href="/blog">blog</a>
                <a href="https://twitter.com/github">Twitter</a>
                <a href="/terms.pdf">User Agreement</a>
            </html>
            """
        )

        crawl("https://github.com", "github.com")

        self.assertIn("https://github.com/features", internal_links)
        self.assertIn("https://docs.github.com", internal_links)
        self.assertIn("https://twitter.com/github", external_links)
        self.assertIn("https://github.com/terms.pdf", file_links)
        self.assertIn("docs.github.com", subdomains)


if __name__ == '__main__':
    unittest.main(verbosity=2)