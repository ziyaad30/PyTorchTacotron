# Default hyperparameters:
class hparams:
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    
    #Dataset Root Directory
    dataset_root = 'Speech'
    
    # text, you may want to use "english_cleaners" or "phoneme_cleaners"
    cleaners='phoneme_cleaners'

    # Audio:
    num_mels = 80
    num_freq = 1025
    sample_rate = 22050
    frame_length_ms = 50
    frame_shift_ms = 12.5
    preemphasis = 0.97
    min_level_db = -100
    ref_level_db = 20

    # Model:
    # TODO: add more configurable hparams
    outputs_per_step = 5
    padding_idx = None
    use_memory_mask = False

    # Data loader
    pin_memory = True
    num_workers = 0

    # Training:
    batch_size = 16
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    initial_learning_rate = 0.002
    decay_learning_rate = True
    run_steps = 500000
    weight_decay = 0.0
    clip_thresh = 1.0
    
    # Logging
    log_interval = 100
    
    # Evaluate
    eval_interval = 500
    
    # Save
    checkpoint_dir = 'ckpt'
    checkpoint_interval = 200

    # Eval:
    max_iters = 10
    griffin_lim_iters = 20
    power = 1.5              # Power to raise magnitudes to prior to Griffin-Lim
    
    # Sample infer text
    sample_text = 'This is a cloned voice for TTS, made with Tacotron 2.'
