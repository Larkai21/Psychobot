"""Configuration file for the psy_supabase program.

This module contains all configurable parameters for the application.
Modify these settings to change application behavior without code changes.
"""

# Default response generation settings
DEFAULT_TOPIC = "supportive_listening"
DEFAULT_EMOTION = "concern"
DEFAULT_APPROACH = "empathy_validation"
DEFAULT_THEME = "general_support"

# Model configurations
TEXT_GENERATING_MODEL = "rasyosef/Phi-1_5-Instruct-v0.1"  # Beware templates should be adjusted regarding this model
TOXIC_CLASSIFICATION_MODEL = "facebook/roberta-hate-speech-dynabench-r4-target"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 🔧 Fix: Backward compatibility alias
TOXICITY_MODEL = TOXIC_CLASSIFICATION_MODEL

# Pain Point Detection Configuration
PAIN_POINT_DETECTION = {
    "similarity_threshold": 0.6,
    "min_occurrences": 2,
    "time_window_days": 30,
    "chunking": {
        "include_full_question": True,
        "min_chunk_length": 4,
        "max_chunks_per_question": 8,
        "remove_redundant_chunks": True,
        "extract_noun_phrases": True,
        "extract_emotional_phrases": True,
        "extract_temporal_phrases": True,
        "extract_action_phrases": True,
        "early_exit_on_perfect_match": True,
        "log_only_best_matches": True,
    },
}

# Database Configuration
DATABASE_CONFIG = {
    "interaction_batch_size": 100,
    "max_sessions_per_user": 50,
    "session_timeout_days": 7,
}

# Response Generation Configuration
RESPONSE_CONFIG = {
    "max_response_length": 500,
    "temperature": 0.7,
    "max_context_tokens": 2048,
}

# Toxicity Detection Configuration
TOXICITY_CONFIG = {
    "threshold": 0.8,
    "check_user_input": True,
    "check_generated_response": True,
}

# CUDA/GPU Performance Configuration
CUDA_CONFIG = {
    "cleanup_threshold": 15,
    "cleanup_time_threshold": 600,
    "random_cleanup_probability": 0.02,
    "force_cpu_fallback": False,
    "lazy_model_loading": True,
    "model_cache_enabled": True,
    "max_cached_models": 4,
    "torch_cuda_empty_cache": True,
    "torch_cuda_synchronize": True,
    "gc_collect_frequency": 1,
    "mixed_precision": True,
    "optimized_attention": True,
    "compile_models": True,
    "quantization_8bit": False,
    "quantization_4bit": False,
    "device_selection": "auto",
    "multi_gpu_strategy": "single",
    "gpu_memory_fraction": 0.85,
    "monitor_memory_usage": True,
    "log_device_info": True,
    "memory_profiling": False,
    "warn_on_memory_pressure": True,
}

# RAG Processing Configuration
RAG_CONFIG = {
    "similarity_threshold": 0.7,
    "max_knowledge_chars": 500,
    "max_conversation_exchanges": 2,
    "vector_cache_enabled": True,
    "similarity_cache_timeout": 60,
    "deduplication_window": 5,
    "max_cache_entries": 10,
    "enable_conversation_context": True,
    "enable_knowledge_context": True,
    "enable_hot_topics": True,
    "enable_associative_memory": True,
    "log_similarity_searches": True,
    "log_cache_hits": True,
    "log_context_sizes": True,
}

# Meta-Reflection Configuration
META_REFLECTION_CONFIG = {
    "enabled": True,
    "reflection_probability": 0.3,
    "min_pattern_threshold": 2,
    "max_reflection_length": 150,
    "thematic_recurrence_threshold": 3,
    "emotional_trajectory_days": 7,
    "pattern_bridging_threshold": 2,
    "respect_clinical_alerts": True,
    "patient_only_access": True,
    "professional_tone_required": True,
    "cache_analysis_results": True,
    "analysis_cache_timeout": 300,
    "max_cached_analyses": 20,
    "log_reflection_generation": True,
    "log_pattern_detection": True,
    "log_safety_overrides": True,
}