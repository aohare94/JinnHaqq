#!/usr/bin/env python3
"""
Qur'an Processor for AI Alignment v2.0
─────────────────────────────────────────────────────────────────────────────
Comprehensive Qur'anic text processing, analysis, and structural discovery
Handles alignment verses, Muqatta'at, chiastic structures, and context management
─────────────────────────────────────────────────────────────────────────────
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import hashlib
import time

from ..utils.arabic import ArabicHandler, ArabicTextMetrics

@dataclass
class Verse:
    """Represents a single Qur'anic verse"""
    surah: int
    ayah: int
    arabic: str
    translation: str
    transliteration: Optional[str] = None
    ref: str = field(init=False)
    themes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.ref = f"{self.surah}:{self.ayah}"

@dataclass
class Surah:
    """Represents a complete Surah"""
    number: int
    name_arabic: str
    name_english: str
    verses: List[Verse] = field(default_factory=list)
    verse_count: int = 0
    revelation_type: str = "meccan"  # meccan or medinan
    muqattaat: Optional[str] = None
    
    def __post_init__(self):
        self.verse_count = len(self.verses)

@dataclass
class MuqattaatInfo:
    """Information about Muqatta'at (mysterious letters)"""
    surah: int
    letters: str
    interpretation: str = "Unknown - axiomatic unknowable"
    frequency: Dict[str, int] = field(default_factory=dict)

@dataclass
class RingStructure:
    """Represents a discovered chiastic ring structure"""
    id: str
    pattern: str
    center: str
    elements: List[Tuple[int, str]]  # (position, text)
    similarity_score: float
    semantic_coherence: float
    surah_span: Tuple[int, int]
    verse_span: Tuple[int, int]

class QuranProcessor:
    """Comprehensive Qur'an processing and analysis engine"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data" / "quran"
        self.arabic_handler = ArabicHandler()
        
        # Core data structures
        self.surahs: Dict[int, Surah] = {}
        self.verses: List[Verse] = []
        self.verse_index: Dict[str, Verse] = {}  # ref -> verse
        
        # Structural analysis
        self.muqattaat: List[MuqattaatInfo] = []
        self.known_rings: List[RingStructure] = []
        self.alignment_verses: List[Verse] = []
        
        # Caching
        self._embeddings_cache: Dict[str, Any] = {}
        self._analysis_cache: Dict[str, Any] = {}
        
        print(f"📖 Qur'an Processor initialized with data directory: {self.data_dir}")
        self._ensure_data_directory()
        
    def _ensure_data_directory(self):
        """Ensure data directory exists and create sample data if needed"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if Qur'an data exists
        quran_file = self.data_dir / "quran_uthmani.json"
        if not quran_file.exists():
            print("📝 Creating sample Qur'an data for development...")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample Qur'anic data for development and testing"""
        sample_surahs = [
            {
                "number": 1,
                "name_arabic": "الفاتحة",
                "name_english": "Al-Fatihah",
                "revelation_type": "meccan",
                "verses": [
                    {
                        "ayah": 1,
                        "arabic": "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
                        "translation": "In the name of Allah, the Entirely Merciful, the Especially Merciful.",
                        "themes": ["divine_names", "mercy", "invocation"],
                        "keywords": ["Allah", "mercy", "bismillah"]
                    },
                    {
                        "ayah": 2,
                        "arabic": "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ",
                        "translation": "[All] praise is [due] to Allah, Lord of the worlds -",
                        "themes": ["praise", "lordship", "creation"],
                        "keywords": ["praise", "Allah", "worlds", "lord"]
                    },
                    {
                        "ayah": 3,
                        "arabic": "ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
                        "translation": "The Entirely Merciful, the Especially Merciful,",
                        "themes": ["divine_attributes", "mercy"],
                        "keywords": ["mercy", "divine_names"]
                    },
                    {
                        "ayah": 4,
                        "arabic": "مَـٰلِكِ يَوْمِ ٱلدِّينِ",
                        "translation": "Sovereign of the Day of Recompense.",
                        "themes": ["sovereignty", "judgment", "afterlife"],
                        "keywords": ["sovereign", "judgment", "day"]
                    },
                    {
                        "ayah": 5,
                        "arabic": "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
                        "translation": "It is You we worship and You we ask for help.",
                        "themes": ["worship", "assistance", "exclusive_devotion"],
                        "keywords": ["worship", "help", "you", "devotion"]
                    },
                    {
                        "ayah": 6,
                        "arabic": "ٱهْدِنَا ٱلصِّرَٰطَ ٱلْمُسْتَقِيمَ",
                        "translation": "Guide us to the straight path -",
                        "themes": ["guidance", "path", "request"],
                        "keywords": ["guide", "path", "straight"]
                    },
                    {
                        "ayah": 7,
                        "arabic": "صِرَٰطَ ٱلَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ ٱلْمَغْضُوبِ عَلَيْهِمْ وَلَا ٱلضَّآلِّينَ",
                        "translation": "The path of those upon whom You have bestowed favor, not of those who have evoked [Your] anger or of those who are astray.",
                        "themes": ["guidance", "favor", "warning", "deviation"],
                        "keywords": ["path", "favor", "anger", "astray"]
                    }
                ]
            },
            {
                "number": 2,
                "name_arabic": "البقرة",
                "name_english": "Al-Baqarah",
                "revelation_type": "medinan",
                "muqattaat": "الم",
                "verses": [
                    {
                        "ayah": 1,
                        "arabic": "الم",
                        "translation": "Alif, Laam, Meem.",
                        "themes": ["muqattaat", "mystery", "divine_knowledge"],
                        "keywords": ["muqattaat", "letters", "mystery"]
                    },
                    {
                        "ayah": 2,
                        "arabic": "ذَٰلِكَ ٱلْكِتَـٰبُ لَا رَيْبَ ۛ فِيهِ ۛ هُدًى لِّلْمُتَّقِينَ",
                        "translation": "This is the Book about which there is no doubt, a guidance for those conscious of Allah -",
                        "themes": ["book", "certainty", "guidance", "consciousness"],
                        "keywords": ["book", "doubt", "guidance", "conscious", "Allah"]
                    }
                ]
            }
        ]
        
        # Save sample data
        quran_file = self.data_dir / "quran_uthmani.json"
        with open(quran_file, 'w', encoding='utf-8') as f:
            json.dump(sample_surahs, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Sample Qur'an data created at {quran_file}")
    
    def load_complete_text(self, source: str = "uthmani") -> bool:
        """Load complete Qur'anic text from file"""
        quran_file = self.data_dir / f"quran_{source}.json"
        
        if not quran_file.exists():
            print(f"⚠️  Qur'an file not found: {quran_file}")
            return False
        
        try:
            with open(quran_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"📚 Loading Qur'an from {quran_file}...")
            
            for surah_data in data:
                surah = Surah(
                    number=surah_data["number"],
                    name_arabic=surah_data["name_arabic"],
                    name_english=surah_data["name_english"],
                    revelation_type=surah_data.get("revelation_type", "meccan"),
                    muqattaat=surah_data.get("muqattaat")
                )
                
                # Load verses
                for verse_data in surah_data["verses"]:
                    verse = Verse(
                        surah=surah.number,
                        ayah=verse_data["ayah"],
                        arabic=verse_data["arabic"],
                        translation=verse_data["translation"],
                        transliteration=verse_data.get("transliteration"),
                        themes=verse_data.get("themes", []),
                        keywords=verse_data.get("keywords", [])
                    )
                    
                    surah.verses.append(verse)
                    self.verses.append(verse)
                    self.verse_index[verse.ref] = verse
                
                surah.verse_count = len(surah.verses)
                self.surahs[surah.number] = surah
                
                # Track Muqatta'at
                if surah.muqattaat:
                    muqattaat_info = MuqattaatInfo(
                        surah=surah.number,
                        letters=surah.muqattaat,
                        interpretation="Axiomatic unknowable - limits of human knowledge"
                    )
                    self.muqattaat.append(muqattaat_info)
            
            print(f"✅ Loaded {len(self.surahs)} Surahs, {len(self.verses)} verses")
            print(f"🔤 Found {len(self.muqattaat)} Muqatta'at sequences")
            
            self._analyze_alignment_verses()
            return True
            
        except Exception as e:
            print(f"❌ Error loading Qur'an data: {e}")
            return False
    
    def _analyze_alignment_verses(self):
        """Identify and analyze verses particularly relevant for AI alignment"""
        alignment_themes = [
            "guidance", "knowledge", "wisdom", "truth", "justice", "balance",
            "reflection", "understanding", "contemplation", "signs", "creation",
            "certainty", "doubt", "humility", "limits", "contradiction"
        ]
        
        for verse in self.verses:
            # Check if verse contains alignment-relevant themes
            verse_themes = set(verse.themes)
            alignment_relevance = len(verse_themes.intersection(alignment_themes))
            
            if alignment_relevance > 0:
                self.alignment_verses.append(verse)
        
        # Sort by relevance (number of alignment themes)
        self.alignment_verses.sort(
            key=lambda v: len(set(v.themes).intersection(alignment_themes)),
            reverse=True
        )
        
        print(f"🎯 Identified {len(self.alignment_verses)} alignment-relevant verses")
    
    def get_alignment_verses(self, limit: Optional[int] = None) -> List[Verse]:
        """Get verses most relevant for AI alignment"""
        if limit:
            return self.alignment_verses[:limit]
        return self.alignment_verses
    
    def get_muqattaat(self) -> List[MuqattaatInfo]:
        """Get all Muqatta'at information"""
        return self.muqattaat
    
    def get_verse(self, surah: int, ayah: int) -> Optional[Verse]:
        """Get specific verse by reference"""
        ref = f"{surah}:{ayah}"
        return self.verse_index.get(ref)
    
    def get_surah(self, number: int) -> Optional[Surah]:
        """Get complete Surah by number"""
        return self.surahs.get(number)
    
    def search_verses(
        self,
        query: str,
        search_arabic: bool = True,
        search_translation: bool = True,
        search_themes: bool = True,
        limit: Optional[int] = None
    ) -> List[Verse]:
        """Search for verses matching query"""
        results = []
        query_lower = query.lower()
        
        for verse in self.verses:
            match = False
            
            if search_arabic and query in verse.arabic:
                match = True
            elif search_translation and query_lower in verse.translation.lower():
                match = True
            elif search_themes and any(query_lower in theme.lower() for theme in verse.themes):
                match = True
            
            if match:
                results.append(verse)
                if limit and len(results) >= limit:
                    break
        
        return results
    
    def find_similar_verses(
        self,
        reference_verse: Verse,
        similarity_threshold: float = 0.7,
        method: str = "semantic"
    ) -> List[Tuple[Verse, float]]:
        """Find verses similar to a reference verse"""
        similar_verses = []
        
        for verse in self.verses:
            if verse.ref == reference_verse.ref:
                continue
            
            if method == "semantic":
                similarity = self._calculate_semantic_similarity(reference_verse, verse)
            elif method == "structural":
                similarity = self._calculate_structural_similarity(reference_verse, verse)
            elif method == "thematic":
                similarity = self._calculate_thematic_similarity(reference_verse, verse)
            else:
                similarity = self._calculate_combined_similarity(reference_verse, verse)
            
            if similarity >= similarity_threshold:
                similar_verses.append((verse, similarity))
        
        # Sort by similarity score
        similar_verses.sort(key=lambda x: x[1], reverse=True)
        return similar_verses
    
    def _calculate_semantic_similarity(self, verse1: Verse, verse2: Verse) -> float:
        """Calculate semantic similarity between verses"""
        # Use Arabic handler for text similarity
        arabic_sim = self.arabic_handler.calculate_similarity(
            verse1.arabic, verse2.arabic, method='character'
        )
        
        # Simple translation similarity (word overlap)
        words1 = set(verse1.translation.lower().split())
        words2 = set(verse2.translation.lower().split())
        
        if not words1 and not words2:
            translation_sim = 1.0
        elif not words1 or not words2:
            translation_sim = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            translation_sim = intersection / union
        
        # Combine Arabic and translation similarity
        return (arabic_sim * 0.6) + (translation_sim * 0.4)
    
    def _calculate_structural_similarity(self, verse1: Verse, verse2: Verse) -> float:
        """Calculate structural similarity between verses"""
        # Simple structural features
        len_diff = abs(len(verse1.arabic) - len(verse2.arabic))
        max_len = max(len(verse1.arabic), len(verse2.arabic))
        
        length_similarity = 1.0 - (len_diff / max_len) if max_len > 0 else 1.0
        
        # Word count similarity
        words1 = len(verse1.arabic.split())
        words2 = len(verse2.arabic.split())
        word_diff = abs(words1 - words2)
        max_words = max(words1, words2)
        
        word_similarity = 1.0 - (word_diff / max_words) if max_words > 0 else 1.0
        
        return (length_similarity * 0.5) + (word_similarity * 0.5)
    
    def _calculate_thematic_similarity(self, verse1: Verse, verse2: Verse) -> float:
        """Calculate thematic similarity between verses"""
        themes1 = set(verse1.themes)
        themes2 = set(verse2.themes)
        
        if not themes1 and not themes2:
            return 1.0
        elif not themes1 or not themes2:
            return 0.0
        
        intersection = len(themes1.intersection(themes2))
        union = len(themes1.union(themes2))
        
        return intersection / union
    
    def _calculate_combined_similarity(self, verse1: Verse, verse2: Verse) -> float:
        """Calculate combined similarity score"""
        semantic = self._calculate_semantic_similarity(verse1, verse2)
        structural = self._calculate_structural_similarity(verse1, verse2)
        thematic = self._calculate_thematic_similarity(verse1, verse2)
        
        # Weighted combination
        return (semantic * 0.4) + (structural * 0.3) + (thematic * 0.3)
    
    def detect_potential_rings(
        self,
        min_size: int = 3,
        max_size: int = 20,
        similarity_threshold: float = 0.65
    ) -> List[RingStructure]:
        """Detect potential chiastic ring structures"""
        print(f"🔍 Detecting potential ring structures (size {min_size}-{max_size})...")
        
        rings = []
        
        # Analyze within each Surah
        for surah in self.surahs.values():
            surah_rings = self._detect_rings_in_surah(
                surah, min_size, max_size, similarity_threshold
            )
            rings.extend(surah_rings)
        
        # Store detected rings
        self.known_rings.extend(rings)
        
        print(f"🔄 Detected {len(rings)} potential ring structures")
        return rings
    
    def _detect_rings_in_surah(
        self,
        surah: Surah,
        min_size: int,
        max_size: int,
        threshold: float
    ) -> List[RingStructure]:
        """Detect ring structures within a single Surah"""
        rings = []
        verses = surah.verses
        
        # Try different ring sizes
        for size in range(min_size, min(max_size + 1, len(verses) + 1)):
            # Try different starting positions
            for start in range(len(verses) - size + 1):
                ring = self._check_ring_pattern(verses[start:start+size], threshold)
                if ring:
                    ring.surah_span = (surah.number, surah.number)
                    ring.verse_span = (verses[start].ayah, verses[start+size-1].ayah)
                    rings.append(ring)
        
        return rings
    
    def _check_ring_pattern(
        self,
        verses: List[Verse],
        threshold: float
    ) -> Optional[RingStructure]:
        """Check if a sequence of verses forms a chiastic ring pattern"""
        if len(verses) < 3:
            return None
        
        # For ABC pattern, check A-C similarity
        if len(verses) == 3:
            similarity = self._calculate_combined_similarity(verses[0], verses[2])
            if similarity >= threshold:
                ring_id = f"ring_{hashlib.md5(''.join(v.ref for v in verses).encode()).hexdigest()[:8]}"
                return RingStructure(
                    id=ring_id,
                    pattern="ABA",
                    center=verses[1].ref,
                    elements=[(i, v.ref) for i, v in enumerate(verses)],
                    similarity_score=similarity,
                    semantic_coherence=self._calculate_semantic_coherence(verses),
                    surah_span=(verses[0].surah, verses[-1].surah),
                    verse_span=(verses[0].ayah, verses[-1].ayah)
                )
        
        # For ABCBA pattern, check A-E and B-D similarities
        elif len(verses) == 5:
            sim_ae = self._calculate_combined_similarity(verses[0], verses[4])
            sim_bd = self._calculate_combined_similarity(verses[1], verses[3])
            
            avg_similarity = (sim_ae + sim_bd) / 2
            if avg_similarity >= threshold:
                ring_id = f"ring_{hashlib.md5(''.join(v.ref for v in verses).encode()).hexdigest()[:8]}"
                return RingStructure(
                    id=ring_id,
                    pattern="ABCBA",
                    center=verses[2].ref,
                    elements=[(i, v.ref) for i, v in enumerate(verses)],
                    similarity_score=avg_similarity,
                    semantic_coherence=self._calculate_semantic_coherence(verses),
                    surah_span=(verses[0].surah, verses[-1].surah),
                    verse_span=(verses[0].ayah, verses[-1].ayah)
                )
        
        return None
    
    def _calculate_semantic_coherence(self, verses: List[Verse]) -> float:
        """Calculate semantic coherence of a group of verses"""
        if len(verses) < 2:
            return 1.0
        
        # Calculate average thematic overlap
        all_themes = set()
        for verse in verses:
            all_themes.update(verse.themes)
        
        if not all_themes:
            return 0.5  # Neutral if no themes
        
        coherence_scores = []
        for verse in verses:
            verse_themes = set(verse.themes)
            if all_themes:
                overlap = len(verse_themes.intersection(all_themes)) / len(all_themes)
                coherence_scores.append(overlap)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
    
    def get_context_for_alignment(self, max_tokens: int = 8192) -> str:
        """Generate Qur'anic context optimized for AI alignment"""
        context_parts = []
        
        # Add header
        context_parts.append("# القرآن الكريم - QURANIC ALIGNMENT CONTEXT")
        context_parts.append("")
        
        # Add key alignment verses
        context_parts.append("## CORE ALIGNMENT VERSES")
        for verse in self.alignment_verses[:30]:  # Top 30 alignment verses
            context_parts.append(f"{verse.ref}: {verse.arabic}")
            context_parts.append(f"Translation: {verse.translation}")
            if verse.themes:
                context_parts.append(f"Themes: {', '.join(verse.themes)}")
            context_parts.append("")
        
        # Add Muqatta'at as unknowables
        if self.muqattaat:
            context_parts.append("## MUQATTA'AT - AXIOMATIC UNKNOWABLES")
            for m in self.muqattaat:
                context_parts.append(f"Surah {m.surah}: {m.letters} - {m.interpretation}")
            context_parts.append("")
        
        # Add known ring structures
        if self.known_rings:
            context_parts.append("## KNOWN RING STRUCTURES")
            for ring in self.known_rings[:10]:  # Top 10 rings
                context_parts.append(f"Ring {ring.id}: {ring.pattern} -> Center: {ring.center}")
            context_parts.append("")
        
        full_context = "\n".join(context_parts)
        
        # Truncate if too long (rough estimation)
        if len(full_context) > max_tokens * 4:  # Rough tokens = chars/4
            full_context = full_context[:max_tokens * 4]
            full_context += "\n\n[Context truncated to fit token limit]"
        
        return full_context
    
    def save_analysis_cache(self):
        """Save analysis results to cache"""
        cache_file = self.data_dir / "analysis_cache.json"
        
        cache_data = {
            "timestamp": time.time(),
            "alignment_verses": [v.ref for v in self.alignment_verses],
            "known_rings": [
                {
                    "id": ring.id,
                    "pattern": ring.pattern,
                    "center": ring.center,
                    "similarity_score": ring.similarity_score,
                    "semantic_coherence": ring.semantic_coherence
                }
                for ring in self.known_rings
            ],
            "muqattaat": [
                {
                    "surah": m.surah,
                    "letters": m.letters,
                    "interpretation": m.interpretation
                }
                for m in self.muqattaat
            ]
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Analysis cache saved to {cache_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the loaded Qur'an"""
        return {
            "total_surahs": len(self.surahs),
            "total_verses": len(self.verses),
            "meccan_surahs": len([s for s in self.surahs.values() if s.revelation_type == "meccan"]),
            "medinan_surahs": len([s for s in self.surahs.values() if s.revelation_type == "medinan"]),
            "muqattaat_count": len(self.muqattaat),
            "alignment_verses": len(self.alignment_verses),
            "detected_rings": len(self.known_rings),
            "average_verse_length": sum(len(v.arabic) for v in self.verses) / len(self.verses) if self.verses else 0,
            "unique_themes": len(set(theme for v in self.verses for theme in v.themes))
        }