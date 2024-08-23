# Broad notes on usage

- The data-generating process is in folder *dgp*
- To run, simply use `python train.py` 
- The script uses Hydra and the train config can be located in folder *config*
- Some basic evals to check grammaticality of generations have been implemented in *evals*
- There is currently only free generation task in this codebase. To add more, check out places with *NOTE* in the code.

# Modifications
- `CORR` means that I tried to debug something
- We pass a language name and a config dict instead of a set of hyperparameters.
    - Language is one of `['english', 'expr', 'dyck']`
    - Hyperparams and grammar for `english` are augmented with `n_conjunctions` (number of conjunctions), `p_conjunctions` (probability of introducing a conjunction), `n_prepositions` (number of prepositions), `relative_clauses` (whether or not relative adjectives and adverbs can occur), and `transitivity` (whether or not transitive and intransitive verbs are separated).
    - Hyperparams for `expr` are `n_digits` (number of digits), `n_ops` (number of operations), and `bracket` (whether or not there are brackets)
    - Hyperparams for `dyck` are `n_brackets` (number of different types of brackets) and `p_nest` (probability that a sequence will be nested)
- Default values for all are encoded in `config/train/conf.yaml`. Note that `config/debug` doesn't reflect any of the changes.
- Documentation modified accordingly.

# SAEs
- All SAE-related code is in `sae/`, except the argument parsing and top-level function call, which are in `train-sae.py`.
- Training the SAE makes use of the following hyperparameters, treated as command-line arguments to `train-sae.py`. For default values, please refer to the code.
    - model_dir: The path to the directory that contains the GPT model whose activations form the training data.
    - ckpt: The filename of the GPT model whose activations form the training data.
    - layer_name: The layer of the model whose outputs we want to disentangle. Options are:
        - `wte` [embedding layer]
        - `wpe` [positional encoding layer]
        - `attn{n}` [n-th layer's attention; n = 0, 1]
        - `mlp{n}` [n-th layer's mlp; n = 0, 1]
        - `res{n}` [n-th layer]
        - `ln_f` [final layer-norm before the LM-head]
    - batch_size: The batch size of the dataloader whose inputs generate the activations for the SAE.
    - exp_factor: The expansion factor of the SAE (the ratio of the hidden size to the input size).
    - alpha: If we want to use L1-regularization, the coefficient of the L1 norm of the latent in the loss function.
    - k: If we want to use top-k-regularization, the number of latent dimensions that are kept nonzero.
    - sparsemax: Whether or not to use a KDS encoder.
    - pre_bias: Whether or not we want to use a learnable bias that is subtracted before the encoder and added after the decoder.
    - norm: Whether we want to normalize the input, the decoder columns, the reconstruction, or any combination thereof.
    - lr: The learning rate of the SAE. Used in an Adam optimizer.
    - train_iters: The number of iterations to train on.
    - val_iters: The number of iterations to validate on.
    - val_interval. The number of iterations after which we validate.
    - patience: If training loss increases (not stays the same) for `patience` consecutive epochs, training is stopped.
    - val_patience: The same, but for validation loss.
    - config: The path to the file (in the model directory) that specifies the parameters for the data-generation process. By default, the same parameters used to train the model are used.
- All of the above are optional arguments except `model_dir`, `ckpt`, and `layer_name`.
- Exactly one of `alpha`, `k` and `sparsemax` must be provided (a required mutually exclusive group of args).
- At most one of `patience` and `val_patience` must be provided (an optional mutually exclusive group of args).
- The dataset and GPT model are loaded according to the saved config file in the `model_dir`.
- Trained SAEs are saved in a subdirectory `sae_{i}` (depending on how many SAEs have been saved previously for the same model), which contains a `config.json` with the above arguments and a `model.pth` file.

# Model and SAE Checkpoints
The following sweeps are done over the given ranges, and also over `exp_factor` in [1, 2, 4, 8].  
The other hyperparameters are fixed; `batch_size` is 128, `lr` $10^{-5}$, `train_iters` 5000, `val_interval` 50.

## Expr
- `results/scratch/12owob2t`: Model trained on prefix Expr.
    - `txuzoop6` (sweep-1i): {alpha: [$10^{-3}$, ..., 100], pre_bias: false, norm: ''}
    - `ouk2auft` (sweep-1ii): {k: [8, 16, 32, 64, 128], pre_bias: false, norm: ''}
    - `0k4sitg1` (sweep-1iii): {sparsemax: [no_kds, recon_dist, dist], pre_bias: false, norm: [input, recon, input+recon]}
    - `4op503ga` (sweep-1iv): {alpha: [$10^{-3}$, ..., 100], pre_bias: false, norm: [input, input+recon]}
    - `jmsz85d0` (sweep-1v): {k: [8, 16, 32, 64, 128], pre_bias: false, norm: [input, input+recon]}
    - `9zw7ax5b` (sweep-1vi): {alpha: [$10^{-3}$, ..., 100], pre_bias: true, norm: 'input+dec'}
    - `yexoe4rc` (sweep-1vii): {k: [8, 16, 32, 64, 128], pre_bias: true, norm: 'input+dec'}
- `results/scratch/ivq8uspe`: Model trained on postfix Expr.

## English
- `results/scratch/3v4gwdfk`: Model trained on English (no prepositions, no relative clauses, p_conj 0.25 for nouns, 0.15 for verbs).
    - `pa35mcy8` (sweep-2i): {alpha: [$10^{-3}$, ..., 100], pre_bias: false, norm: ''}
    - `rsydzm4a` (sweep-2ii): {k: [8, 16, 32, 64, 128], pre_bias: false, norm: ''}
    - `k8v87yur` (sweep-2iii): {alpha: [$10^{-3}$, ..., 100], pre_bias: true, norm: 'input+dec'}
    - `ewwdogo3` (sweep-2iv): {k: [8, 16, 32, 64, 128], pre_bias: true, norm: 'input+dec'}
- `results/scratch/vx8j11gp`: Model trained on English with transitivity and no other variations.
    - `wovrxleh` (sweep-3i): {alpha: [$10^{-3}$, ..., 100], pre_bias: false, norm: ''}
    - `1k0zvaa1` (sweep-3ii): {k: [8, 16, 32, 64, 128], pre_bias: false, norm: ''}
- `results/scratch/9rts35mx`: Model trained on English with adverbial and adjectival relative clauses.
    - `5xrp9ao5` (sweep-4i): {alpha: [$10^{-3}$, ..., 100], pre_bias: false, norm: ''}
    - `vlzsuuw2` (sweep-4ii): {k: [8, 16, 32, 64, 128], pre_bias: false, norm: ''}
- `results/scratch/bcb19qnd`: Model trained on English with adverbial and adjectival prepositional phrases (3 prepositions).
    - `982f370u` (sweep-5i): {alpha: [$10^{-3}$, ..., 100], pre_bias: false, norm: ''}
    - `jz5ehe50` (sweep-5ii): {k: [8, 16, 32, 64, 128], pre_bias: false, norm: ''}
- `results/scratch/ktm5d2gn`: Model trained on English with the probability of conjunctions in NPs and VPs set to 0.3.

## Dyck
- `results/scratch/cpyib3ss`: Model trained on Dyck.
    - `z659wka4` (sweep-6i): {alpha: [$10^{-3}$, ..., 100], pre_bias: false, norm: ''}
    - `d6e3v8kc` (sweep-6ii): {k: [8, 16, 32, 64, 128], pre_bias: false, norm: ''}
    - `rmghx4cb` (sweep-6iii): {alpha: [$10^{-3}$, ..., 100], pre_bias: true, norm: 'input+dec'}
    - `jketab8t` (sweep-6iv): {k: [8, 16, 32, 64, 128], pre_bias: true, norm: 'input+dec'}
    - `6yia6guy` (sweep-6v): {alpha: [$10^{-3}$, ..., 100], pre_bias: false, norm: '', config: {p_nest: 0.3}}
    - `m8gp41ku` (sweep-6vi): {k: [8, 16, 32, 64, 128], pre_bias: false, norm: '', config: {p_nest: 0.3}}
- `results/scratch/87bnn1o6`: Model trained on Dyck with the probability of nesting set to 30%. Average depth about 5.
    - `4t092q2g` (sweep-7i): {alpha: [$10^{-3}$, ..., 100], pre_bias: False, norm: ''}
    - `lm32lb8a` (sweep-7ii): {k: [8, 16, 32, 64, 128], pre_bias: False, norm: ''}
    - `y75dol3u` (sweep-7iii): {alpha: [$10^{-3}$, ..., 100], pre_bias: True, norm: 'input+dec'}
    - `68o9f3fm` (sweep-7iv): {k: [8, 16, 32, 64, 128], pre_bias: True, norm: 'input+dec'}
