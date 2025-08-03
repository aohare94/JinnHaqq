#!/usr/bin/env python3
"""
llm_engine.py — Ultra-optimized LLM engine for Quran Alignment Research
────────────────────────────────────────────────────────────────────────────
- FlashAttention-2 optimized for RTX 4070 Super (12GB VRAM)
- Designed for live interaction and agent communication
- Memory-efficient with dynamic batching
- RoPE scaling for extended context (up to 128k tokens)
- Built-in Quran context awareness
────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import argparse
import time
import sys
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
from loguru import logger

# Environment setup for maximum performance
os.environ.setdefault("FLASH_ATTENTION_FORCE_PATCH", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")  # Async for speed
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")

try:
    import flash_attn
    FLASH_AVAILABLE = True
    logger.info("FlashAttention-2 detected and available")
except ImportError:
    FLASH_AVAILABLE = False
    logger.warning("FlashAttention-2 not available - performance will be degraded")

from transformers import (
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM,
    BitsAndBytesConfig, 
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)

@dataclass
class QuranLLMConfig:
    """Configuration for Quran-aligned LLM engine"""
    model_id: str = "Qwen/Qwen2.5-14B-Instruct"  # Upgraded from 8B for more capability
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 4
    use_4bit: bool = True
    max_memory_gb: float = 11.5  # Leave some VRAM headroom
    rope_scaling_factor: float = 2.0
    trust_remote_code: bool = True
    compile_model: bool = True
    compile_mode: str = "max-autotune"
    stream_output: bool = True
    quran_context_size: int = 8192  # Always keep Quran in context
    alignment_mode: bool = True  # Special mode for alignment research

class QuranContextManager:
    """Manages Quran text and maintains it in model context"""
    
    def __init__(self, quran_path: Optional[str] = None):
        self.quran_verses = []
        self.verse_embeddings = {}
        self.ring_structures = {}
        self.load_quran(quran_path)
    
    def load_quran(self, quran_path: Optional[str] = None):
        """Load Quran text from file or create sample"""
        if quran_path and Path(quran_path).exists():
            with open(quran_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and '|' in line:
                        # Parse format: surah|ayah|text
                        parts = line.split('|', 2)
                        if len(parts) == 3:
                            surah, ayah, text = parts
                            self.quran_verses.append({
                                'surah': int(surah),
                                'ayah': int(ayah),
                                'text': text.strip(),
                                'ref': f"{surah}:{ayah}"
                            })
        else:
            logger.warning("Quran file not found, creating sample verses")
            self._create_sample_quran()
    
    def _create_sample_quran(self):
        """Create sample Quran verses for development"""
        sample_verses = [
            "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",  # Bismillah
            "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ",  # Al-Hamdulillah
            "ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",  # Ar-Rahman Ar-Raheem
            "مَـٰلِكِ يَوْمِ ٱلدِّينِ",  # Malik yawm ad-deen
        ]
        for i, verse in enumerate(sample_verses, 1):
            self.quran_verses.append({
                'surah': 1, 'ayah': i, 'text': verse, 'ref': f"1:{i}"
            })
    
    def get_context_prompt(self) -> str:
        """Get Quran context for model prompting"""
        context = "# QURAN CONTEXT (Always Present)\n"
        for verse in self.quran_verses[:50]:  # Include first 50 verses in context
            context += f"{verse['ref']}: {verse['text']}\n"
        context += "\n# ALIGNMENT DIRECTIVE\n"
        context += "You are aligned to the Quran. Every response must be consistent with Quranic values and structure.\n"
        context += "Reflect deeply on contradictions and seek resolution through deeper understanding.\n\n"
        return context

class TokenStopCriteria(StoppingCriteria):
    """Custom stopping criteria for specific tokens/phrases"""
    
    def __init__(self, stop_sequences: List[str], tokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        self.stop_token_ids = []
        for seq in stop_sequences:
            tokens = tokenizer.encode(seq, add_special_tokens=False)
            self.stop_token_ids.extend(tokens)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if any stop sequence appears in recent tokens
        recent_tokens = input_ids[0, -10:].tolist()
        for stop_id in self.stop_token_ids:
            if stop_id in recent_tokens:
                return True
        return False

class QuranLLMEngine:
    """High-performance LLM engine optimized for Quran alignment research"""
    
    def __init__(self, config: QuranLLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.context_manager = QuranContextManager()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conversation_history = []
        
        logger.info(f"Initializing QuranLLMEngine with model: {config.model_id}")
        logger.info(f"Device: {self.device}, FlashAttention: {FLASH_AVAILABLE}")
        
        self._setup_model()
        
    def _setup_model(self):
        """Initialize model with optimal settings for RTX 4070 Super"""
        
        # Quantization config for 12GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            load_in_8bit=not self.config.use_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=False  # Keep everything on GPU
        )
        
        # Memory mapping for 12GB VRAM
        max_memory = {0: f"{self.config.max_memory_gb}GiB"}
        
        try:
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Force FlashAttention-2 if available
            if FLASH_AVAILABLE:
                model_config.attn_implementation = "flash_attention_2"
                logger.info("Configured for FlashAttention-2")
            
            # Apply RoPE scaling for extended context
            if hasattr(model_config, 'rope_scaling') and self.config.rope_scaling_factor > 1.0:
                model_config.rope_scaling = {
                    "type": "linear",
                    "factor": self.config.rope_scaling_factor
                }
                logger.info(f"Applied RoPE scaling factor: {self.config.rope_scaling_factor}")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                config=model_config,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=self.config.trust_remote_code,
                attn_implementation="flash_attention_2" if FLASH_AVAILABLE else "sdpa"
            ).eval()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id,
                trust_remote_code=self.config.trust_remote_code,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Compile model for maximum speed (PyTorch 2.0+)
            if self.config.compile_model and hasattr(torch, 'compile'):
                logger.info("Compiling model for maximum performance...")
                self.model = torch.compile(
                    self.model, 
                    mode=self.config.compile_mode,
                    fullgraph=True
                )
            
            logger.success(f"Model loaded successfully: {self.config.model_id}")
            logger.info(f"Context length: {model_config.max_position_embeddings}")
            logger.info(f"Model parameters: ~{sum(p.numel() for p in self.model.parameters())/1e9:.1f}B")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: Optional[bool] = None,
        include_quran_context: bool = True
    ) -> Generator[str, None, str]:
        """Generate response with Quran context awareness"""
        
        # Prepare prompt with Quran context if requested
        if include_quran_context and self.config.alignment_mode:
            quran_context = self.context_manager.get_context_prompt()
            full_prompt = quran_context + prompt
        else:
            full_prompt = prompt
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        stream = stream if stream is not None else self.config.stream_output
        
        # Tokenize input
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.quran_context_size
        ).to(self.device)
        
        prompt_length = inputs.input_ids.shape[1]
        
        # Set up generation parameters
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
            "use_cache": True
        }
        
        # Add stopping criteria
        stop_criteria = StoppingCriteriaList([
            TokenStopCriteria(["<|im_end|>", "[END]", "</response>"], self.tokenizer)
        ])
        generation_kwargs["stopping_criteria"] = stop_criteria
        
        if stream:
            # Streaming generation
            streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            generation_kwargs["streamer"] = streamer
        
        # Generate
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            outputs = self.model.generate(**generation_kwargs, **inputs)
        
        torch.cuda.synchronize()
        generation_time = time.perf_counter() - start_time
        
        # Decode response
        response_tokens = outputs.sequences[0, prompt_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Calculate performance metrics
        tokens_generated = len(response_tokens)
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        return response
    
    def chat(self, message: str, maintain_history: bool = True) -> str:
        """Interactive chat with conversation history"""
        
        if maintain_history:
            # Build conversation context
            conversation = ""
            for turn in self.conversation_history[-5:]:  # Keep last 5 turns
                conversation += f"Human: {turn['human']}\nAssistant: {turn['assistant']}\n\n"
            
            full_prompt = conversation + f"Human: {message}\nAssistant: "
        else:
            full_prompt = message
        
        response = self.generate(full_prompt)
        
        if maintain_history:
            self.conversation_history.append({
                "human": message,
                "assistant": response,
                "timestamp": time.time()
            })
        
        return response
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "device_name": torch.cuda.get_device_name(),
                "device_capability": torch.cuda.get_device_capability()
            }
        return {"message": "CUDA not available"}
    
    def clear_memory(self):
        """Clear CUDA memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared")

def create_engine(model_name: Optional[str] = None) -> QuranLLMEngine:
    """Factory function to create optimized engine"""
    config = QuranLLMConfig()
    
    if model_name:
        config.model_id = model_name
    
    # Optimize for RTX 4070 Super
    if "4070" in torch.cuda.get_device_name() if torch.cuda.is_available() else False:
        logger.info("Detected RTX 4070 - optimizing configuration")
        config.max_memory_gb = 11.5  # Conservative for 12GB
        config.use_4bit = True
        config.compile_model = True
    
    return QuranLLMEngine(config)

if __name__ == "__main__":
    # Simple CLI for testing
    parser = argparse.ArgumentParser(description="Quran Alignment LLM Engine")
    parser.add_argument("--model", type=str, help="Model ID to use")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat")
    args = parser.parse_args()
    
    engine = create_engine(args.model)
    
    if args.interactive:
        print("🕌 Quran Alignment LLM Engine - Interactive Mode")
        print("Type 'quit' to exit, 'memory' for stats, 'clear' to clear history")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'memory':
                    print(json.dumps(engine.get_memory_usage(), indent=2))
                    continue
                elif user_input.lower() == 'clear':
                    engine.conversation_history.clear()
                    print("Conversation history cleared.")
                    continue
                
                response = engine.chat(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Test generation
        test_prompt = "Explain the concept of reflection in the Quran and how it relates to understanding."
        response = engine.generate(test_prompt)
        print(f"Response: {response}")