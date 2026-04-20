"""Text normalization for TTS input, with Polish-specific handling."""

import re
from abc import ABC, abstractmethod


class TextNormalizer(ABC):
    """Abstract text normalizer. Subclass for language-specific rules."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize text for TTS consumption."""


class PolishTextNormalizer(TextNormalizer):
    """Polish text normalization: numbers, abbreviations, dates, currency.

    CosyVoice2's BPE tokenizer handles raw characters, but explicit
    normalization of numbers and abbreviations produces much better
    pronunciation for Polish.

    Requires: num2words (pip install num2words)
    """

    # Common Polish abbreviations -> full forms
    ABBREVIATIONS = {
        "ul.": "ulica",
        "al.": "aleja",
        "pl.": "plac",
        "os.": "osiedle",
        "nr": "numer",
        "nr.": "numer",
        "dr": "doktor",
        "dr.": "doktor",
        "mgr": "magister",
        "mgr.": "magister",
        "inż.": "inżynier",
        "prof.": "profesor",
        "doc.": "docent",
        "godz.": "godzina",
        "min.": "minut",
        "sek.": "sekund",
        "tys.": "tysięcy",
        "mln": "milionów",
        "mld": "miliardów",
        "wg": "według",
        "np.": "na przykład",
        "tj.": "to jest",
        "m.in.": "między innymi",
        "itd.": "i tak dalej",
        "itp.": "i tym podobne",
        "tzw.": "tak zwany",
        "ok.": "około",
        "r.": "roku",
        "w.": "wiek",
        "im.": "imienia",
        "św.": "święty",
        "śp.": "świętej pamięci",
        "ds.": "do spraw",
        "pn.": "poniedziałek",
        "wt.": "wtorek",
        "śr.": "środa",
        "czw.": "czwartek",
        "pt.": "piątek",
        "sob.": "sobota",
        "ndz.": "niedziela",
    }

    # Currency symbol -> num2words currency code
    CURRENCY_CODES = {
        "zł": "PLN",
        "PLN": "PLN",
        "EUR": "EUR",
        "USD": "USD",
    }

    MONTHS_GENITIVE = {
        1: "stycznia",
        2: "lutego",
        3: "marca",
        4: "kwietnia",
        5: "maja",
        6: "czerwca",
        7: "lipca",
        8: "sierpnia",
        9: "września",
        10: "października",
        11: "listopada",
        12: "grudnia",
    }

    def __init__(self):
        try:
            import num2words
            self._num2words = num2words
        except ImportError:
            raise ImportError(
                "num2words is required for PolishTextNormalizer. "
                "Install with: pip install num2words"
            )

    def normalize(self, text: str) -> str:
        text = self._expand_abbreviations(text)
        text = self._normalize_times(text)
        text = self._normalize_dates(text)
        text = self._normalize_currency(text)
        text = self._numbers_to_words(text)
        text = self._clean_whitespace(text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        # Sort by length descending so "m.in." matches before "m."
        for abbrev, expansion in sorted(
            self.ABBREVIATIONS.items(), key=lambda x: -len(x[0])
        ):
            # Require word boundary (or start of string) before the abbreviation
            # and whitespace/punctuation/end after it
            pattern = re.compile(
                r"(?<!\w)" + re.escape(abbrev) + r"(?=\s|$|,|;)",
                re.IGNORECASE,
            )
            text = pattern.sub(expansion, text)
        return text

    def _normalize_times(self, text: str) -> str:
        """Convert 14:30 -> czternasta trzydzieści."""

        def _time_to_words(match):
            h, m = int(match.group(1)), int(match.group(2))
            h_word = self._num2words.num2words(h, lang="pl")
            if m == 0:
                return h_word
            m_word = self._num2words.num2words(m, lang="pl")
            return f"{h_word} {m_word}"

        return re.sub(r"\b(\d{1,2}):(\d{2})\b", _time_to_words, text)

    def _normalize_dates(self, text: str) -> str:
        """Convert 20.04.2026 -> dwudziestego kwietnia dwa tysiące dwadzieścia sześć."""

        def _date_to_words(match):
            day, month, year = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            day_word = self._num2words.num2words(day, lang="pl", to="ordinal")
            # Genitive form approximation: append "ego" if not already
            if not day_word.endswith("ego"):
                day_word = day_word.rstrip("y") + "ego"
            month_word = self.MONTHS_GENITIVE.get(month, str(month))
            year_word = self._num2words.num2words(year, lang="pl")
            return f"{day_word} {month_word} {year_word}"

        return re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", _date_to_words, text)

    def _normalize_currency(self, text: str) -> str:
        """Convert '25 zł' -> 'dwadzieścia pięć złotych' with correct declension.

        Uses num2words currency mode which handles Polish złoty/złote/złotych
        and grosz/grosze/groszy declension automatically.
        """
        for symbol, currency_code in self.CURRENCY_CODES.items():
            pattern = re.compile(
                r"(\d+(?:[.,]\d+)?)\s*" + re.escape(symbol) + r"(?!\w)"
            )
            for match in pattern.finditer(text):
                num_str = match.group(1).replace(",", ".")
                try:
                    amount = float(num_str)
                except ValueError:
                    continue
                words = self._num2words.num2words(
                    amount, lang="pl", to="currency", currency=currency_code
                )
                # Remove ", zero groszy" / ", zero centów" suffix for whole amounts
                words = re.sub(r",\s*zero\s+\w+$", "", words)
                text = text.replace(match.group(0), words)
        return text

    def _numbers_to_words(self, text: str) -> str:
        """Convert remaining standalone numbers to Polish words."""

        def _num_to_word(match):
            num_str = match.group(0)
            try:
                if "." in num_str or "," in num_str:
                    num = float(num_str.replace(",", "."))
                else:
                    num = int(num_str)
                return self._num2words.num2words(num, lang="pl")
            except (ValueError, OverflowError):
                return num_str

        return re.sub(r"\b\d+(?:[.,]\d+)?\b", _num_to_word, text)

    def _clean_whitespace(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()
