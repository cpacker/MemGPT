DETERMINISTIC = {
      'max_new_tokens': 250,
      'do_sample': False,
      'temperature': 0,
      'top_p': 0,
      'typical_p': 1,
      'repetition_penalty': 1.18,
      'repetition_penalty_range': 0,
      'encoder_repetition_penalty': 1,
      'top_k': 1,
      'min_length': 0,
      'no_repeat_ngram_size': 0,
      'num_beams': 1,
      'penalty_alpha': 0,
      'length_penalty': 1,
      'early_stopping': False,
      'guidance_scale': 1,
      'negative_prompt': '',
      'seed': -1,
      'add_bos_token': True,
      'stopping_strings': [
        '\nUSER:',
        '\nASSISTANT:',
        # '\n' +
        # '</s>',
        # '<|',
        # '\n#',
        # '\n\n\n',
      ],
      'truncation_length': 4096,
      'ban_eos_token': False,
      'skip_special_tokens': True,
      'top_a': 0,
      'tfs': 1,
      'epsilon_cutoff': 0,
      'eta_cutoff': 0,
      'mirostat_mode': 2,
      'mirostat_tau': 4,
      'mirostat_eta': 0.1,
      'use_mancer': False
    }

SIMPLE = {
      'stopping_strings': [
        '\nUSER:',
        '\nASSISTANT:',
        # '\n' +
        # '</s>',
        # '<|',
        # '\n#',
        # '\n\n\n',
      ],
      'truncation_length': 4096,
}