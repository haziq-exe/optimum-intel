# Haziq Tiny-minicpm-o Branch

This README.md goes into details about how I compressed the given MiniCPM-o-2_6.

## Generating Model

### TTS Model

The majority of the compression comes from pruning the TTS model's DVAE. I replaced the 12-layer Encoder/Decoder structure with a lightweight 2-layer implementation. Additionally, I reduced the internal hidden states and convolution channels from 512/256 down to 48 dimensions. These changes can be found in the `modeling.minicpmo.py` file in the model folder.

### Vocabulary Size Changes

I also compressed the model's vocabulary, limiting both text and audio vocabularies to a hard cap of 1,024 tokens. This involved completely reconstructing the tokenizer assets, including the vocab.json, merges.txt, and tokenizer.json files. 

The new vocabulary includes the most critical special tokens (such as <|endoftext|> and <|image|>), and the most frequent tokens from the original set. Consequently, I re-synced the model configuration to align with these new token IDs and filtered the merge rules to fit the reduced scope.

This reduced the tokenizer size from ~6MB to less than 0.5MB

### Layer Configuration Changes

I scaled down the architecture across the text, vision, and audio components. I configured the core LLM backbone to use a hidden size of 48, utilizing only 2 hidden layers and 4 attention heads to minimize parameter count. This compact 48-dimensional configuration was also used in the vision and audio encoders, creating a consistent low-rank bottleneck throughout the system.

To implement this, I had to manually modify the original model source code. The default modeling_minicpmo.py enforces a hard constraint requiring the embed_dim to be at least 128 (attention heads would be equal embed_dim // 128). I modified the source code `modeling.minicpmo.py` to no longer have this requirement. Allowing the model to initialize successfully with dimensions far below the standard architectural floor.

## Validation Script

The validation script includes a small forward pass of the model with a random image and accompanying text asking to describe the image.

Finally, the script also tests the TTS components by passing a dummy tensor through the DVAE decoder. This verifies that the dimensionality reduction was applied correctly and that the new dimension sizes align with the compressed weight


## Github Actions Validation

I configured the workflow yml files of the tests coresponding to minicpmo model to run when commit is published to this branch (not in parallel to avoid HF rate limit). I also updated `test_exporters_cli.py` and `test_quantization.py` files so that the quantized layer number matches the compressed model.