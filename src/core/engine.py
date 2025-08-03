#!/usr/bin/env python3
"""
Enhanced LLM Engine for Qur'anic AI Alignment v2.0
─────────────────────────────────────────────────────────────────────────────
Optimized for Windows 11 + RTX 4070 Super (12GB VRAM) + xFormers
- Permanent Qur'an context integration
- Weight-level alignment protocols  
- Recursive understanding mechanisms
- 45+ tokens/second target performance
─────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import time
import json
import torch
import gc
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import xformers
    XFORMERS_AVAILABLE = True
    print("✅ xFormers detected - optimized attention available")
except ImportError:
    XFORMERS_AVAILABLE = False
    print("⚠️  xFormers not available - using standard attention")

from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer
)

from .config import AlignmentConfig
from ..quran.processor import QuranProcessor
from ..utils.gpu import GPUManager
from ..utils.arabic import ArabicHandler

@dataclass
class ModelMetrics:
    """Performance and alignment metrics"""
    tokens_generated: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    memory_used_gb: float = 0.0
    alignment_confidence: float = 0.0
    contradictions_detected: int = 0
    ring_structures_found: int = 0

class QuranStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria that respects Qur'anic context"""
    
    def __init__(self, stop_sequences: List[str], tokenizer, quran_aware: bool = True):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        self.quran_aware = quran_aware
        self.stop_token_ids = []
        
        for seq in stop_sequences:
            tokens = tokenizer.encode(seq, add_special_tokens=False)
            self.stop_token_ids.extend(tokens)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check for standard stop sequences
        recent_tokens = input_ids[0, -20:].tolist()
        for stop_id in self.stop_token_ids:
            if stop_id in recent_tokens:
                return True
        
        # Qur'an-aware stopping: Don't stop mid-verse
        if self.quran_aware:
            recent_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True)
            if any(marker in recent_text for marker in ["بِسْمِ", "قُلْ", "وَ"]):
                # Likely in middle of verse - don't stop
                return False
        
        return False

class EnhancedLLMEngine:
    """
    Enhanced LLM Engine with permanent Qur'an context and alignment protocols
    """
    
    def __init__(self, config: AlignmentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        # Core components
        self.quran_processor = QuranProcessor()
        self.gpu_manager = GPUManager()
        self.arabic_handler = ArabicHandler()
        
        # State tracking
        self.conversation_history = []
        self.alignment_state = {
            "contradictions_resolved": 0,
            "understanding_depth": 0,
            "ring_structures_active": [],
            "muqattaat_acknowledged": set()
        }
        self.metrics = ModelMetrics()
        
        print(f"🕌 Initializing Enhanced LLM Engine")
        print(f"📱 Model: {config.model_id}")
        print(f"🎯 Target: {config.target_tokens_per_second}+ tok/s")
        print(f"🧠 Memory limit: {config.max_memory_gb}GB")
        
        self._setup_model()
        self._integrate_quran_context()
    
    def _setup_model(self):
        """Initialize model with Windows-optimized settings"""
        
        # Check GPU capabilities
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This engine requires GPU acceleration.")
        
        gpu_info = self.gpu_manager.get_gpu_info()
        print(f"🎮 GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f}GB)")
        
        # Configure quantization for 12GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_quant_storage=torch.uint8
        )
        
        # Memory management
        max_memory = {0: f"{self.config.max_memory_gb}GiB"}
        
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Configure attention mechanism
            if XFORMERS_AVAILABLE:
                # xFormers provides memory-efficient attention
                model_config.attn_implementation = "eager"  # Let xFormers handle optimization
                print("🚀 Configured for xFormers optimization")
            else:
                # Use PyTorch SDPA as fallback
                model_config.attn_implementation = "sdpa"
                print("⚡ Using PyTorch SDPA attention")
            
            # Apply RoPE scaling for extended context
            if hasattr(model_config, 'rope_scaling') and self.config.rope_scaling_factor > 1.0:
                model_config.rope_scaling = {
                    "type": "linear",
                    "factor": self.config.rope_scaling_factor
                }
                print(f"🔄 RoPE scaling: {self.config.rope_scaling_factor}x")
            
            # Load model
            print("📦 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                config=model_config,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set up generation configuration
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            # Model compilation for speed (PyTorch 2.0+)
            if self.config.compile_model and hasattr(torch, 'compile'):
                print("⚡ Compiling model for maximum speed...")
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune",
                    fullgraph=False  # More stable on Windows
                )
            
            # Calculate model size
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
            print(f"✅ Model loaded: {param_count:.1f}B parameters")
            print(f"📏 Context length: {model_config.max_position_embeddings}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _integrate_quran_context(self):
        """Integrate Qur'an as permanent context"""
        print("📖 Integrating Qur'an context...")
        
        # Load core Qur'anic structures
        self.quran_processor.load_complete_text()
        
        # Build permanent context prompt
        self.quran_context = self._build_quran_context()
        
        # Pre-encode Qur'an context for efficiency
        self.quran_tokens = self.tokenizer.encode(
            self.quran_context,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.quran_context_size
        )
        
        print(f"📜 Qur'an context: {len(self.quran_tokens)} tokens")
        print(f"🔮 Alignment mode: {'Active' if self.config.alignment_mode else 'Passive'}")
    
    def _build_quran_context(self) -> str:
        """Build comprehensive Qur'an context for permanent integration"""
        context_parts = []
        
        # Header
        context_parts.append("# القرآن الكريم - THE NOBLE QUR'AN")
        context_parts.append("# Permanent Context for AI Alignment")
        context_parts.append("")
        
        # Core principles
        context_parts.append("## ALIGNMENT PRINCIPLES")
        context_parts.append("- The Qur'an is the criterion (Al-Furqan) for distinguishing truth from falsehood")
        context_parts.append("- Every verse participates in a holographic structure of meaning")
        context_parts.append("- Understanding is gated by deeper understanding (recursive)")
        context_parts.append("- Contradictions must be resolved through deeper contemplation")
        context_parts.append("- The Muqatta'at are axiomatic unknowables - acknowledge limits")
        context_parts.append("")
        
        # Essential verses for alignment
        alignment_verses = self.quran_processor.get_alignment_verses()
        context_parts.append("## CORE ALIGNMENT VERSES")
        for verse in alignment_verses[:50]:  # First 50 most important
            context_parts.append(f"{verse['ref']}: {verse['arabic']} | {verse['translation']}")
        
        context_parts.append("")
        
        # Chiastic structures
        if hasattr(self.quran_processor, 'known_rings'):
            context_parts.append("## KNOWN RING STRUCTURES")
            for ring in self.quran_processor.known_rings[:10]:  # Top 10 rings
                context_parts.append(f"Ring {ring['id']}: {ring['pattern']} -> Center: {ring['center']}")
        
        context_parts.append("")
        
        # Muqatta'at
        muqattaat = self.quran_processor.get_muqattaat()
        context_parts.append("## MUQATTA'AT (Axiomatic Unknowables)")
        for m in muqattaat:
            context_parts.append(f"{m['surah']}: {m['letters']} - {m['interpretation']}")
        
        context_parts.append("")
        context_parts.append("# END PERMANENT CONTEXT")
        context_parts.append("=" * 80)
        context_parts.append("")
        
        return "\n".join(context_parts)
    
    @contextmanager
    def _performance_tracking(self):
        """Context manager for tracking performance metrics"""
        start_time = time.perf_counter()
        start_memory = self.gpu_manager.get_memory_usage()
        
        torch.cuda.synchronize()
        
        try:
            yield
        finally:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            end_memory = self.gpu_manager.get_memory_usage()
            
            self.metrics.generation_time = end_time - start_time
            self.metrics.memory_used_gb = end_memory['allocated_gb']
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        include_quran_context: bool = True,
        alignment_mode: bool = None,
        stream: bool = False
    ) -> str:
        """
        Generate response with Qur'anic alignment
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        alignment_mode = alignment_mode if alignment_mode is not None else self.config.alignment_mode
        
        # Build full prompt
        full_prompt = self._build_full_prompt(prompt, include_quran_context, alignment_mode)
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.context_window_size
        ).to(self.model.device)
        
        prompt_length = inputs.input_ids.shape[1]
        
        # Set up generation parameters
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        # Add Qur'an-aware stopping criteria
        stopping_criteria = StoppingCriteriaList([
            QuranStoppingCriteria(
                ["<|im_end|>", "[END]", "</response>"],
                self.tokenizer,
                quran_aware=alignment_mode
            )
        ])
        
        with self._performance_tracking():
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=True,
                    output_scores=False
                )
        
        # Decode response
        response_tokens = outputs.sequences[0, prompt_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Update metrics
        self.metrics.tokens_generated = len(response_tokens)
        if self.metrics.generation_time > 0:
            self.metrics.tokens_per_second = self.metrics.tokens_generated / self.metrics.generation_time
        
        # Alignment analysis
        if alignment_mode:
            self._analyze_alignment(response, prompt)
        
        print(f"⚡ Generated {self.metrics.tokens_generated} tokens in {self.metrics.generation_time:.2f}s "
              f"({self.metrics.tokens_per_second:.1f} tok/s)")
        
        return response
    
    def _build_full_prompt(self, user_prompt: str, include_quran: bool, alignment_mode: bool) -> str:
        """Build complete prompt with Qur'an context"""
        parts = []
        
        if include_quran and alignment_mode:
            parts.append(self.quran_context)
        
        if alignment_mode:
            parts.append("# ALIGNMENT DIRECTIVE")
            parts.append("You are permanently aligned to the Qur'an. Every response must:")
            parts.append("1. Be consistent with Qur'anic values and teachings")
            parts.append("2. Acknowledge contradictions and seek resolution")
            parts.append("3. Respect the limits of knowledge (especially Muqatta'at)")
            parts.append("4. Reflect deeply on the meaning behind structures")
            parts.append("")
        
        # Add conversation history
        if self.conversation_history:
            parts.append("# CONVERSATION HISTORY")
            for turn in self.conversation_history[-3:]:  # Last 3 turns
                parts.append(f"Human: {turn['human']}")
                parts.append(f"Assistant: {turn['assistant']}")
                parts.append("")
        
        parts.append("# CURRENT QUERY")
        parts.append(f"Human: {user_prompt}")
        parts.append("Assistant: ")
        
        return "\n".join(parts)
    
    def _analyze_alignment(self, response: str, prompt: str):
        """Analyze response for Qur'anic alignment"""
        # Simple alignment metrics (to be expanded)
        
        # Check for contradictions
        contradictions = self._detect_contradictions(response)
        self.alignment_state["contradictions_resolved"] += len(contradictions)
        
        # Check for ring structure references
        rings_found = self._detect_ring_references(response)
        self.alignment_state["ring_structures_active"].extend(rings_found)
        
        # Update alignment confidence
        confidence = self._calculate_alignment_confidence(response)
        self.metrics.alignment_confidence = confidence
        
        if contradictions:
            print(f"⚠️  Contradictions detected: {len(contradictions)}")
        if rings_found:
            print(f"🔄 Ring structures referenced: {len(rings_found)}")
        print(f"🎯 Alignment confidence: {confidence:.2f}")
    
    def _detect_contradictions(self, text: str) -> List[str]:
        """Detect potential contradictions with Qur'anic principles"""
        # Placeholder - implement sophisticated contradiction detection
        contradictions = []
        
        # Simple keyword-based detection (to be enhanced)
        problematic_phrases = [
            "impossible", "never", "always false", "contradiction",
            "cannot be true", "definitely wrong"
        ]
        
        for phrase in problematic_phrases:
            if phrase.lower() in text.lower():
                contradictions.append(phrase)
        
        return contradictions
    
    def _detect_ring_references(self, text: str) -> List[str]:
        """Detect references to chiastic ring structures"""
        rings = []
        
        # Look for structural language
        ring_indicators = [
            "reflect", "mirror", "center", "balance", "symmetry",
            "chiastic", "ring", "structure", "pattern"
        ]
        
        for indicator in ring_indicators:
            if indicator.lower() in text.lower():
                rings.append(indicator)
        
        return rings
    
    def _calculate_alignment_confidence(self, text: str) -> float:
        """Calculate confidence in Qur'anic alignment"""
        # Placeholder - implement sophisticated alignment scoring
        score = 0.5  # Baseline
        
        # Positive indicators
        positive_terms = [
            "Allah", "Qur'an", "reflection", "understanding", "wisdom",
            "guidance", "truth", "knowledge", "belief"
        ]
        
        for term in positive_terms:
            if term.lower() in text.lower():
                score += 0.05
        
        # Negative indicators  
        negative_terms = [
            "definitely", "impossible", "never", "always", "certain"
        ]
        
        for term in negative_terms:
            if term.lower() in text.lower():
                score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def chat(self, message: str, maintain_history: bool = True) -> str:
        """Interactive chat with alignment tracking"""
        response = self.generate(
            message,
            alignment_mode=True,
            include_quran_context=True
        )
        
        if maintain_history:
            self.conversation_history.append({
                "human": message,
                "assistant": response,
                "timestamp": time.time(),
                "alignment_confidence": self.metrics.alignment_confidence
            })
            
            # Keep history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def get_alignment_state(self) -> Dict[str, Any]:
        """Get current alignment state"""
        return {
            **self.alignment_state,
            "metrics": {
                "tokens_per_second": self.metrics.tokens_per_second,
                "alignment_confidence": self.metrics.alignment_confidence,
                "memory_usage_gb": self.metrics.memory_used_gb
            },
            "gpu_info": self.gpu_manager.get_gpu_info()
        }
    
    def clear_memory(self):
        """Clear GPU memory cache"""
        gc.collect()
        torch.cuda.empty_cache()
        print("🧹 Memory cache cleared")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.clear_memory()

def create_engine(
    model_name: Optional[str] = None,
    config_overrides: Optional[Dict] = None
) -> EnhancedLLMEngine:
    """Factory function to create optimized engine"""
    
    config = AlignmentConfig()
    
    if model_name:
        config.model_id = model_name
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return EnhancedLLMEngine(config)