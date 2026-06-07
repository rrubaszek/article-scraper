from google import genai

class ArticleGenerator:
    def __init__(self, client: genai.Client, MODEL):
        self.client = client
        self.MODEL = MODEL

    def build_prompt(self, topic):
        return f"""
        Napisz artykuł prasowy w języku polskim.

        Temat: {topic}

        Styl:
        - neutralny styl dziennikarski
        - jak w portalu informacyjnym
        - brak opinii
        - brak list punktowanych

        Wymagania:
        - 500-800 słów
        - struktura: nagłówek + lead + treść artykułu
        - realistyczny styl newsowy
        - bez wzmianki o AI

        Zwróć tylko treść artykułu.
        """
        
    def generate_article(self, topic):
        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=self.build_prompt(topic)
        )

        return {
            "url": None,
            "title": topic,
            "date": None,
            "text": response.text.strip(),
            "source": "gemini_ai"
        }