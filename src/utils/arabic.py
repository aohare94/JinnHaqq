#!/usr/bin/env python3
"""
Arabic Text Handler for Qur'anic AI Alignment
─────────────────────────────────────────────────────────────────────────────
Comprehensive Arabic text processing with Qur'anic considerations
Handles normalization, diacritics, text direction, and structural analysis
─────────────────────────────────────────────────────────────────────────────
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import json

# Arabic Unicode ranges and characters
ARABIC_LETTERS = set(range(0x0621, 0x0650))  # Basic Arabic letters
ARABIC_DIACRITICS = set(range(0x064B, 0x0660))  # Diacritical marks
ARABIC_EXTENDED = set(range(0x0671, 0x06D4))  # Extended Arabic
ARABIC_SUPPLEMENT = set(range(0x0750, 0x077F))  # Arabic Supplement

# Common Qur'anic text patterns
VERSE_MARKERS = ["۝", "۞", "۩", "﴾", "﴿"]
SURAH_MARKERS = ["سُورَة", "سورة"]
BASMALA = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"

@dataclass
class ArabicTextMetrics:
    """Metrics for Arabic text analysis"""
    character_count: int
    word_count: int
    letter_count: int
    diacritic_count: int
    unique_letters: Set[str]
    text_direction: str
    has_quranic_markers: bool

class ArabicHandler:
    """Comprehensive Arabic text processing for Qur'anic content"""
    
    def __init__(self):
        self.diacritic_map = self._build_diacritic_map()
        self.letter_forms = self._build_letter_forms()
        self.stop_words = self._load_arabic_stop_words()
        self.root_patterns = self._build_root_patterns()
        
    def _build_diacritic_map(self) -> Dict[str, str]:
        """Build mapping of diacritical marks"""
        return {
            '\u064B': 'FATHATAN',     # ً
            '\u064C': 'DAMMATAN',     # ٌ
            '\u064D': 'KASRATAN',     # ٍ
            '\u064E': 'FATHA',        # َ
            '\u064F': 'DAMMA',        # ُ
            '\u0650': 'KASRA',        # ِ
            '\u0651': 'SHADDA',       # ّ
            '\u0652': 'SUKUN',        # ْ
            '\u0653': 'MADDAH',       # ٓ
            '\u0654': 'HAMZA_ABOVE',  # ٔ
            '\u0655': 'HAMZA_BELOW',  # ٕ
            '\u0656': 'SUBSCRIPT_ALEF', # ٖ
            '\u0657': 'INVERTED_DAMMA', # ٗ
            '\u0658': 'MARK_NOON_GHUNNA', # ٘
            '\u0670': 'SUPERSCRIPT_ALEF', # ٰ
        }
    
    def _build_letter_forms(self) -> Dict[str, Dict[str, str]]:
        """Build mapping of Arabic letter forms"""
        # Simplified mapping - in practice, use a comprehensive Arabic shaping library
        return {
            'ا': {'isolated': 'ا', 'initial': 'ا', 'medial': 'ا', 'final': 'ا'},
            'ب': {'isolated': 'ب', 'initial': 'بـ', 'medial': 'ـبـ', 'final': 'ـب'},
            'ت': {'isolated': 'ت', 'initial': 'تـ', 'medial': 'ـتـ', 'final': 'ـت'},
            # Add more letters as needed
        }
    
    def _load_arabic_stop_words(self) -> Set[str]:
        """Load common Arabic stop words"""
        return {
            'من', 'إلى', 'عن', 'في', 'على', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
            'الذي', 'التي', 'اللذان', 'اللتان', 'الذين', 'اللواتي', 'اللائي',
            'ما', 'لا', 'لم', 'لن', 'إن', 'أن', 'كان', 'كانت', 'يكون', 'تكون'
        }
    
    def _build_root_patterns(self) -> List[str]:
        """Build common Arabic root patterns"""
        return [
            'فعل', 'فعال', 'مفعل', 'فاعل', 'مفعول', 'فعيل', 'فعول',
            'تفعيل', 'استفعال', 'انفعال', 'افتعال', 'تفاعل'
        ]
    
    def normalize_text(self, text: str, preserve_diacritics: bool = True) -> str:
        """Normalize Arabic text while preserving Qur'anic integrity"""
        if not text:
            return ""
        
        # Normalize Unicode composition
        text = unicodedata.normalize('NFKC', text)
        
        # Handle common character variations
        text = text.replace('ي', 'ی')  # Standardize Yeh
        text = text.replace('ك', 'ک')  # Standardize Kaf
        text = text.replace('ة', 'ۃ')  # Standardize Teh Marbuta for Qur'anic text
        
        # Normalize Alef variations
        text = re.sub(r'[آأإٱ]', 'ا', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if not preserve_diacritics:
            text = self.remove_diacritics(text)
        
        return text
    
    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritical marks"""
        if not text:
            return ""
        
        # Remove all diacritical marks
        diacritic_pattern = '[\u064B-\u0652\u0653-\u0658\u0670]'
        return re.sub(diacritic_pattern, '', text)
    
    def extract_diacritics(self, text: str) -> List[Tuple[int, str, str]]:
        """Extract diacritical marks with their positions"""
        diacritics = []
        for i, char in enumerate(text):
            if char in self.diacritic_map:
                diacritics.append((i, char, self.diacritic_map[char]))
        return diacritics
    
    def analyze_text(self, text: str) -> ArabicTextMetrics:
        """Comprehensive analysis of Arabic text"""
        if not text:
            return ArabicTextMetrics(0, 0, 0, 0, set(), 'ltr', False)
        
        # Basic counts
        character_count = len(text)
        word_count = len(text.split())
        
        # Arabic-specific counts
        letter_count = 0
        diacritic_count = 0
        unique_letters = set()
        
        for char in text:
            if ord(char) in ARABIC_LETTERS:
                letter_count += 1
                unique_letters.add(char)
            elif ord(char) in ARABIC_DIACRITICS:
                diacritic_count += 1
        
        # Text direction (RTL for Arabic)
        text_direction = 'rtl' if letter_count > 0 else 'ltr'
        
        # Check for Qur'anic markers
        has_quranic_markers = any(marker in text for marker in VERSE_MARKERS)
        
        return ArabicTextMetrics(
            character_count=character_count,
            word_count=word_count,
            letter_count=letter_count,
            diacritic_count=diacritic_count,
            unique_letters=unique_letters,
            text_direction=text_direction,
            has_quranic_markers=has_quranic_markers
        )
    
    def extract_words(self, text: str, remove_stop_words: bool = False) -> List[str]:
        """Extract words from Arabic text"""
        if not text:
            return []
        
        # Split on whitespace and punctuation
        words = re.findall(r'[\u0621-\u06FF]+', text)
        
        if remove_stop_words:
            words = [word for word in words if word not in self.stop_words]
        
        return words
    
    def extract_roots(self, word: str) -> List[str]:
        """Extract potential Arabic roots from a word"""
        if not word or len(word) < 3:
            return []
        
        # Remove diacritics for root extraction
        clean_word = self.remove_diacritics(word)
        
        # Simple root extraction (3-letter combinations)
        potential_roots = []
        
        if len(clean_word) >= 3:
            # Try different 3-letter combinations
            for i in range(len(clean_word) - 2):
                potential_root = clean_word[i:i+3]
                if self._is_valid_root(potential_root):
                    potential_roots.append(potential_root)
        
        return potential_roots
    
    def _is_valid_root(self, root: str) -> bool:
        """Check if a 3-letter combination could be a valid Arabic root"""
        if len(root) != 3:
            return False
        
        # Check if all characters are Arabic letters
        for char in root:
            if ord(char) not in ARABIC_LETTERS:
                return False
        
        # Simple validation - avoid repeated letters or common non-root patterns
        if len(set(root)) < 2:  # Avoid roots with all same letters
            return False
        
        return True
    
    def detect_verse_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """Detect verse boundaries in Qur'anic text"""
        boundaries = []
        
        # Look for verse markers
        for marker in VERSE_MARKERS:
            for match in re.finditer(re.escape(marker), text):
                boundaries.append((match.start(), match.end()))
        
        # Look for numeric verse markers (e.g., ﴿١﴾)
        verse_pattern = r'﴿\d+﴾'
        for match in re.finditer(verse_pattern, text):
            boundaries.append((match.start(), match.end()))
        
        return sorted(boundaries)
    
    def detect_basmala(self, text: str) -> List[int]:
        """Detect Basmala occurrences in text"""
        positions = []
        
        # Exact match
        for match in re.finditer(re.escape(BASMALA), text):
            positions.append(match.start())
        
        # Fuzzy match (without some diacritics)
        simplified_basmala = self.remove_diacritics(BASMALA)
        simplified_text = self.remove_diacritics(text)
        
        for match in re.finditer(re.escape(simplified_basmala), simplified_text):
            positions.append(match.start())
        
        return sorted(set(positions))
    
    def segment_text(self, text: str) -> Dict[str, List[str]]:
        """Segment text into different components"""
        segments = {
            'verses': [],
            'words': [],
            'letters': [],
            'diacritics': []
        }
        
        # Extract verses using markers
        verse_boundaries = self.detect_verse_boundaries(text)
        if verse_boundaries:
            last_end = 0
            for start, end in verse_boundaries:
                if start > last_end:
                    verse_text = text[last_end:start].strip()
                    if verse_text:
                        segments['verses'].append(verse_text)
                last_end = end
            
            # Add final segment
            if last_end < len(text):
                final_verse = text[last_end:].strip()
                if final_verse:
                    segments['verses'].append(final_verse)
        else:
            # No markers found, treat as single verse
            segments['verses'] = [text]
        
        # Extract words
        segments['words'] = self.extract_words(text)
        
        # Extract letters and diacritics
        for char in text:
            if ord(char) in ARABIC_LETTERS:
                segments['letters'].append(char)
            elif ord(char) in ARABIC_DIACRITICS:
                segments['diacritics'].append(char)
        
        return segments
    
    def calculate_similarity(self, text1: str, text2: str, method: str = 'character') -> float:
        """Calculate similarity between two Arabic texts"""
        if not text1 or not text2:
            return 0.0
        
        if method == 'character':
            return self._character_similarity(text1, text2)
        elif method == 'word':
            return self._word_similarity(text1, text2)
        elif method == 'root':
            return self._root_similarity(text1, text2)
        else:
            return self._character_similarity(text1, text2)
    
    def _character_similarity(self, text1: str, text2: str) -> float:
        """Character-level similarity"""
        set1 = set(text1)
        set2 = set(text2)
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _word_similarity(self, text1: str, text2: str) -> float:
        """Word-level similarity"""
        words1 = set(self.extract_words(text1))
        words2 = set(self.extract_words(text2))
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _root_similarity(self, text1: str, text2: str) -> float:
        """Root-level similarity"""
        words1 = self.extract_words(text1)
        words2 = self.extract_words(text2)
        
        roots1 = set()
        roots2 = set()
        
        for word in words1:
            roots1.update(self.extract_roots(word))
        
        for word in words2:
            roots2.update(self.extract_roots(word))
        
        if not roots1 and not roots2:
            return 1.0
        
        intersection = len(roots1.intersection(roots2))
        union = len(roots1.union(roots2))
        
        return intersection / union if union > 0 else 0.0
    
    def format_for_display(self, text: str, direction: str = 'rtl') -> str:
        """Format Arabic text for proper display"""
        if not text:
            return ""
        
        # Add RTL override characters if needed
        if direction == 'rtl':
            return f"\u202B{text}\u202C"  # RTL override
        else:
            return text
    
    def validate_quranic_text(self, text: str) -> Dict[str, any]:
        """Validate if text appears to be authentic Qur'anic content"""
        validation = {
            'is_arabic': False,
            'has_diacritics': False,
            'has_quranic_markers': False,
            'has_basmala': False,
            'confidence_score': 0.0,
            'warnings': []
        }
        
        if not text:
            validation['warnings'].append("Empty text")
            return validation
        
        metrics = self.analyze_text(text)
        
        # Check if text is primarily Arabic
        validation['is_arabic'] = metrics.letter_count > 0
        
        # Check for diacritics (common in Qur'anic text)
        validation['has_diacritics'] = metrics.diacritic_count > 0
        
        # Check for Qur'anic markers
        validation['has_quranic_markers'] = metrics.has_quranic_markers
        
        # Check for Basmala
        validation['has_basmala'] = len(self.detect_basmala(text)) > 0
        
        # Calculate confidence score
        score = 0.0
        if validation['is_arabic']:
            score += 0.4
        if validation['has_diacritics']:
            score += 0.2
        if validation['has_quranic_markers']:
            score += 0.3
        if validation['has_basmala']:
            score += 0.1
        
        validation['confidence_score'] = score
        
        # Add warnings
        if not validation['is_arabic']:
            validation['warnings'].append("Text does not appear to be Arabic")
        if score < 0.5:
            validation['warnings'].append("Text may not be authentic Qur'anic content")
        
        return validation
    
    def prepare_for_embeddings(self, text: str, normalize: bool = True) -> str:
        """Prepare Arabic text for embedding models"""
        if not text:
            return ""
        
        if normalize:
            text = self.normalize_text(text, preserve_diacritics=True)
        
        # Add language hint for multilingual models
        return f"ar: {text}"