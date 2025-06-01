import nltk
import unicodedata
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

try:
    stopwords.words('portuguese')
    RSLPStemmer()
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('rslp', quiet=True)

STOP_WORDS_PT = set(stopwords.words('portuguese'))
STEMMER_PT = RSLPStemmer()

def preprocessar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    tokens = texto.split()
    tokens_processados = [STEMMER_PT.stem(palavra) for palavra in tokens if palavra not in STOP_WORDS_PT]
    return " ".join(tokens_processados)

def contar_palavras_lexico(texto_processado: str, lexico_palavras: list) -> int:
    contador = 0
    if isinstance(texto_processado, str):
        palavras_texto = texto_processado.split()
        for palavra_lexico in lexico_palavras:
            if palavra_lexico in palavras_texto:
                contador += 1
    return contador
