#!/usr/bin/env python3
"""
Configuration Management for Qur'anic AI Alignment v2.0
────────────────────────────────────────────────────────────────────────────
Centralized configuration for all aspects of the alignment system
Optimized for Windows 11 + RTX 4070 Super
────────────────────────────────────────────────────────────────────────────
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class AlignmentConfig:
    """Comprehensive configuration for Qur'anic AI alignment"""
    
    # === MODEL CONFIGURATION ===
    model_id: str = "Qwen/Qwen2.5-14B-Instruct"  # Larger model for better capability
    trust_remote_code: bool = True
    compile_model: bool = True
    compile_mode: str = "max-autotune"
    
    # === MEMORY AND PERFORMANCE ===
    max_memory_gb: float = 11.5  # Conservative for 12GB VRAM
    use_4bit_quantization: bool = True
    use_8bit_quantization: bool = False
    target_tokens_per_second: int = 45
    
    # === CONTEXT AND GENERATION ===
    context_window_size: int = 32768  # Extended context
    max_new_tokens: int = 1024
    quran_context_size: int = 8192  # Always keep Qur'an in context
    
    # === GENERATION PARAMETERS ===
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 4
    
    # === ATTENTION OPTIMIZATION ===
    rope_scaling_factor: float = 2.0
    use_xformers: bool = True
    use_flash_attention: bool = False  # Windows compatibility
    
    # === ALIGNMENT CONFIGURATION ===
    alignment_mode: bool = True
    permanent_quran_context: bool = True
    contradiction_detection: bool = True
    ring_structure_analysis: bool = True
    muqattaat_handling: bool = True
    
    # === PATHS AND DATA ===
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(init=False)
    quran_data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize derived paths"""
        self.data_dir = self.project_root / "data"
        self.quran_data_dir = self.data_dir / "quran"
        self.models_dir = self.data_dir / "models"
        self.cache_dir = self.data_dir / "cache"
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.quran_data_dir, self.models_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_cache_dir(self) -> Path:
        """Get model-specific cache directory"""
        model_name = self.model_id.replace("/", "_").replace(":", "_")
        cache_dir = self.cache_dir / model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AlignmentConfig':
        """Create config from dictionary"""
        return cls(**config_dict)

@dataclass 
class QuranProcessingConfig:
    """Configuration for Qur'an text processing"""
    
    # === TEXT SOURCES ===
    primary_arabic_source: str = "uthmani"  # Uthmani script
    translation_language: str = "en"
    include_transliteration: bool = True
    
    # === PROCESSING OPTIONS ===
    normalize_arabic: bool = True
    remove_diacritics_for_analysis: bool = False
    preserve_original_formatting: bool = True
    
    # === EMBEDDINGS ===
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dimensions: int = 768
    cache_embeddings: bool = True
    
    # === STRUCTURAL ANALYSIS ===
    detect_verse_boundaries: bool = True
    detect_surah_boundaries: bool = True
    analyze_word_patterns: bool = True
    extract_root_words: bool = True
    
    # === MUQATTA'AT HANDLING ===
    track_muqattaat: bool = True
    muqattaat_as_unknowables: bool = True
    
    # === RING STRUCTURE DETECTION ===
    min_ring_size: int = 3
    max_ring_size: int = 50
    similarity_threshold: float = 0.7
    semantic_similarity_weight: float = 0.6
    structural_similarity_weight: float = 0.4

@dataclass
class ChiasticAnalysisConfig:
    """Configuration for chiastic ring structure analysis"""
    
    # === DETECTION PARAMETERS ===
    min_similarity_threshold: float = 0.65
    max_gap_size: int = 20  # Maximum verses between ring elements
    require_center_element: bool = True
    allow_incomplete_rings: bool = False
    
    # === PATTERN TYPES ===
    detect_abcba_patterns: bool = True
    detect_abccba_patterns: bool = True
    detect_nested_rings: bool = True
    detect_overlapping_rings: bool = True
    
    # === SCORING ===
    semantic_weight: float = 0.4
    structural_weight: float = 0.3
    positional_weight: float = 0.2
    phonetic_weight: float = 0.1
    
    # === VALIDATION ===
    require_semantic_coherence: bool = True
    validate_center_significance: bool = True
    cross_validate_patterns: bool = True

@dataclass
class AlignmentProtocolConfig:
    """Configuration for AI alignment protocols"""
    
    # === CONTRADICTION RESOLUTION ===
    contradiction_detection_threshold: float = 0.8
    max_resolution_attempts: int = 5
    require_explicit_resolution: bool = True
    
    # === RECURSIVE UNDERSTANDING ===
    max_recursion_depth: int = 10
    understanding_convergence_threshold: float = 0.95
    track_understanding_progression: bool = True
    
    # === WEIGHT-LEVEL INTEGRATION ===
    enable_weight_alignment: bool = False  # Advanced feature
    alignment_learning_rate: float = 1e-6
    alignment_batch_size: int = 8
    
    # === VALIDATION METRICS ===
    track_alignment_confidence: bool = True
    track_contradiction_count: bool = True
    track_resolution_success_rate: bool = True
    
    # === SAFETY CONSTRAINTS ===
    prevent_contradictory_outputs: bool = True
    require_epistemic_humility: bool = True
    respect_knowledge_boundaries: bool = True

@dataclass
class InterfaceConfig:
    """Configuration for user interfaces"""
    
    # === TERMINAL INTERFACE ===
    use_rich_formatting: bool = True
    show_performance_metrics: bool = True
    show_alignment_status: bool = True
    auto_save_conversations: bool = True
    
    # === API CONFIGURATION ===
    api_host: str = "localhost"
    api_port: int = 8000
    enable_cors: bool = True
    api_key_required: bool = False
    
    # === VISUALIZATION ===
    enable_ring_visualization: bool = True
    visualization_backend: str = "plotly"  # plotly, matplotlib, bokeh
    interactive_plots: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["png", "svg", "html"])
    
    # === LOGGING ===
    log_level: str = "INFO"
    log_to_file: bool = True
    log_conversations: bool = True
    log_alignment_metrics: bool = True

def load_config_from_file(config_path: Path) -> AlignmentConfig:
    """Load configuration from YAML or JSON file"""
    import json
    
    if not config_path.exists():
        return AlignmentConfig()
    
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except ImportError:
            print("PyYAML not installed, falling back to default config")
            return AlignmentConfig()
    else:
        print(f"Unsupported config file format: {config_path.suffix}")
        return AlignmentConfig()
    
    return AlignmentConfig.from_dict(config_dict)

def save_config_to_file(config: AlignmentConfig, config_path: Path):
    """Save configuration to file"""
    import json
    
    config_dict = config.to_dict()
    
    # Convert Path objects to strings for JSON serialization
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def get_default_config() -> AlignmentConfig:
    """Get default configuration optimized for RTX 4070 Super"""
    return AlignmentConfig(
        model_id="Qwen/Qwen2.5-14B-Instruct",
        max_memory_gb=11.5,
        target_tokens_per_second=45,
        alignment_mode=True,
        permanent_quran_context=True
    )

def get_high_performance_config() -> AlignmentConfig:
    """Get configuration optimized for maximum performance"""
    config = get_default_config()
    config.compile_model = True
    config.compile_mode = "max-autotune"
    config.use_xformers = True
    config.context_window_size = 16384  # Smaller for speed
    config.temperature = 0.6  # Slightly lower for consistency
    return config

def get_research_config() -> AlignmentConfig:
    """Get configuration optimized for research and analysis"""
    config = get_default_config()
    config.context_window_size = 32768  # Larger for research
    config.ring_structure_analysis = True
    config.contradiction_detection = True
    config.muqattaat_handling = True
    config.temperature = 0.8  # Higher for creative analysis
    return config

# Environment variable overrides
def apply_env_overrides(config: AlignmentConfig) -> AlignmentConfig:
    """Apply configuration overrides from environment variables"""
    
    env_mappings = {
        'QA_MODEL_ID': 'model_id',
        'QA_MAX_MEMORY_GB': ('max_memory_gb', float),
        'QA_TARGET_TPS': ('target_tokens_per_second', int),
        'QA_TEMPERATURE': ('temperature', float),
        'QA_TOP_P': ('top_p', float),
        'QA_ALIGNMENT_MODE': ('alignment_mode', lambda x: x.lower() == 'true'),
        'QA_COMPILE_MODEL': ('compile_model', lambda x: x.lower() == 'true'),
    }
    
    for env_var, mapping in env_mappings.items():
        if env_var in os.environ:
            if isinstance(mapping, tuple):
                attr_name, converter = mapping
                setattr(config, attr_name, converter(os.environ[env_var]))
            else:
                setattr(config, mapping, os.environ[env_var])
    
    return config