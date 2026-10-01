---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:4100000
- loss:CachedMultipleNegativesRankingLoss
widget:
- source_sentence: 'query: aditya tech private limited | plot no:556 8-2-293/82/jiii
    road no:92 jubilee hills hyderabad telangana'
  sentences:
  - 'query: ग्लोबल कंस्ट्रक्शन प्राइवेट लिमिटेड | 470 lucknow null uttar pradesh'
  - 'query: स्मार्ट इंजीनियरिंग प्राइवेट लिमिटेड | h.no 222 shipura meerut up'
  - 'query: ఆదిత్య టెక్ ప్రైవేట్ లిమిటెడ్ | hyderabad telangana plot no:556'
- source_sentence: 'query: lycee medicale services | tourcoing 6 rue denfert rochereau
    hauts-de-france'
  sentences:
  - 'query: centre hospitalier commercants | boulevard du president bordeaux nouvelle-aquitaine'
  - 'query: lycee medicale services services'
  - 'query: uu societe | 5 rue jules de vicq lille hauts-de-france'
- source_sentence: 'query: chandrayagrand.com | up house no-1/2/14 saharanpur cinema
    hall market urf n'
  sentences:
  - 'query: ahmedabadventure.net | b1013 empire business hub science city road nr
    cims hospital sola ahmedabad ahmedabad gujarat'
  - 'query: chandrayagrand.com | up house saharanpur cinema hall market urf n'
  - 'query: lksmi phaundesn stors llp | bazar ajmeri 5918 3rd floor basti harphool
    singh sadar'
- source_sentence: 'query: adinath memorial trust pvt ltd | d/4/92 nehru nagar new
    delhi delhi'
  sentences:
  - 'query: komal komal (india) unity private | rz d - 91 50 feet road nihal vihar
    nangloi new delhi delhi'
  - 'query: adinath memorial trust | d/4/92 nehru nagar new delhi'
  - 'query: arun law chambevs industries llp | ruby 504 sri sairam manor pragatinagar
    road hyderabad tg'
- source_sentence: 'query: inc. red tattoo | 197 dry creek road goodlettsville tn'
  sentences:
  - 'query: 2star.com | cheste city mt whitlash road'
  - 'query: llc soloeon churchill | 675 130th avenue monmouth'
  - 'query: inc. red red tattoo | 197 dry creek rd goodlettsville tn'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained on the noisy_France, noisy_India, script_India, matches_India, script_matches_India, noisy_US and matches_US datasets. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
- **Training Datasets:**
    - noisy_France
    - noisy_India
    - script_India
    - matches_India
    - script_matches_India
    - noisy_US
    - matches_US
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'query: inc. red tattoo | 197 dry creek road goodlettsville tn',
    'query: inc. red red tattoo | 197 dry creek rd goodlettsville tn',
    'query: llc soloeon churchill | 675 130th avenue monmouth',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.9671, -0.0663],
#         [ 0.9671,  1.0000, -0.1135],
#         [-0.0663, -0.1135,  1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Datasets
<details><summary>noisy_France</summary>

#### noisy_France

* Dataset: noisy_France
* Size: 500,000 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            |
  | details | <ul><li>min: 6 tokens</li><li>mean: 17.97 tokens</li><li>max: 34 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 17.42 tokens</li><li>max: 35 tokens</li></ul> |
* Samples:
  | anchor                                                                                                     | positive                                                                                              |
  |:-----------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
  | <code>query: 2014 palet anciens sasu \| 56 bis rue de l’ocean pornic pays de la loire</code>               | <code>query: 2014 palet anciens sasu \| 56 bis r de l’ocean pornc pays de la loire</code>             |
  | <code>query: voisins & cie participations sarl \| 8 rue desbiey la teste de buch nouvelle-aquitaine</code> | <code>query: voisins & cie participations sarl \| 8 rue desbiey la teste de nouvelle-aquitaine</code> |
  | <code>query: 24h fetes sarl \| 9 rue des loisirs dunkerque hauts-de-france</code>                          | <code>query: 24h fetes sarl</code>                                                                    |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 128,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```
</details>
<details><summary>noisy_India</summary>

#### noisy_India

* Dataset: noisy_India
* Size: 500,000 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            |
  | details | <ul><li>min: 7 tokens</li><li>mean: 24.45 tokens</li><li>max: 54 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 23.13 tokens</li><li>max: 57 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                      | positive                                                                                                                    |
  |:----------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|
  | <code>query: effective trading private limited \| no.90- uthukottai tk kannigaipair uthukottai uttukottai tamil nadu</code> | <code>query: effective trading private limited \| tk kannigaipair uthukottai uttukottai tamil nadu no.90- uthukottai</code> |
  | <code>query: wm \| 404 pride sapphire janvi park main road rajkot gujarat</code>                                            | <code>query: wm \| 404 pride sapphire janvi park main rd rajkot gujarat</code>                                              |
  | <code>query: pf private limited services \| b-601 nishat complex nagarwada vadodara vadodara gujarat</code>                 | <code>query: pf private limitted services \| complex nagarwada vadodara vadodara gujarat b-601 nishat</code>                |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 128,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```
</details>
<details><summary>script_India</summary>

#### script_India

* Dataset: script_India
* Size: 300,000 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            |
  | details | <ul><li>min: 11 tokens</li><li>mean: 25.79 tokens</li><li>max: 55 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 26.3 tokens</li><li>max: 56 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                       | positive                                                                                                                                     |
  |:---------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>query: panchkula limited services \| door no b3/388, basement, industrial area, phase 1, sec sixteen,panchkula haryana, hriyana</code> | <code>query: panchkula limited services \| door no b3/388, basement, industrial area, phase 1, sec sixteen,panchkula haryana, हरियाणा</code> |
  | <code>query: incredible royal private limited \| villa no-123, ramachandrapuram, medak, telngan</code>                                       | <code>query: incredible royal private limited \| villa no-123, ramachandrapuram, medak, తెలంగాణ</code>                                       |
  | <code>query: om phuds private limited \| building no 340 thekkedath house kottayam kuravilangad keralam</code>                               | <code>query: ഓം ഫുഡ്സ് പ്രൈവറ്റ് ലിമിറ്റഡ് \| building no 340 thekkedath house kottayam kuravilangad keralam</code>                          |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 128,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```
</details>
<details><summary>matches_India</summary>

#### matches_India

* Dataset: matches_India
* Size: 1,000,000 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                          | negative                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            | string                                                                            |
  | details | <ul><li>min: 12 tokens</li><li>mean: 24.97 tokens</li><li>max: 45 tokens</li></ul> | <ul><li>min: 7 tokens</li><li>mean: 23.99 tokens</li><li>max: 47 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 22.62 tokens</li><li>max: 50 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                     | positive                                                                                                                                        | negative                                                                                                     |
  |:-------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|
  | <code>query: anand infrastructure private limited \| a/101 amrut park bldg no7 khadakpada kalyan west kalyan thane maharashtra</code>      | <code>query: anand infrastructure private limited \| plot 407 a/101 amrut park bldg no7 khadakpada kalyan west kalyan goveli maharashtra</code> | <code>query: anand ventures private limited \| flat no 701 kalyan thane maharashtra</code>                   |
  | <code>query: high trading private limited \| no.69/1 bangalore sadanandanagar benniganahalli karnataka</code>                              | <code>query: high trading private limited \| #69/1 bangalore bengaluru ka</code>                                                                | <code>query: best trading private limited \| bangalore, krnatk, 69, 11th b main, 5th block, jayanagar</code> |
  | <code>query: jmk housing pvt. ltd \| c/o anpurna kumari house no.0 village and tola -malpur panch-lohagir ujiarpur samastipur bihar</code> | <code>query: jmk housoing pvt. ltd \| c/o anpurna kumari, house no.c-0 village and tola -malpur, panch-lohagir, ujiarpur, bihar</code>          | <code>query: qvh \| c/o manish kumar village tola -gopalpur begusarai bihar</code>                           |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 128,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```
</details>
<details><summary>script_matches_India</summary>

#### script_matches_India

* Dataset: script_matches_India
* Size: 300,000 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             |
  | details | <ul><li>min: 11 tokens</li><li>mean: 24.44 tokens</li><li>max: 51 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 26.02 tokens</li><li>max: 55 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                 | positive                                                                                                                                      |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>query: vision systems private limited \| nagla sardar kachoura hathras uttar pradesh</code>                                                      | <code>query: vision systems private ltd \| nagla sardar kachoura, hathras, उत्तर प्रदेश</code>                                                |
  | <code>query: bunty communication private limited \| no.45 46 47 ptr nagar chennai to kumbakonam road vadakuthu kurinjipadi cuddalore tamil nadu</code> | <code>query: bunty communication \| no.45 , 46, 47, ptr nagar, chennai to kumbakonam road, vadakuthu kurinjipadi, cuddalore, தமிழ்நாடு</code> |
  | <code>query: metro distilleries private limited \| h.n 1272 mahadev johari ki gali gopal ji ka rasta jaipur rajasthan</code>                           | <code>query: metro distilleries private \| h.n. 1272, mahadev johari, ki gali, goal ji ka rasta, jaipur, राजस्थान</code>                      |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 128,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```
</details>
<details><summary>noisy_US</summary>

#### noisy_US

* Dataset: noisy_US
* Size: 500,000 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            |
  | details | <ul><li>min: 6 tokens</li><li>mean: 16.89 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 16.69 tokens</li><li>max: 31 tokens</li></ul> |
* Samples:
  | anchor                                                                               | positive                                                                              |
  |:-------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | <code>query: trusted defense products \| 2217 20th street clarkstonc ity wa</code>   | <code>query: trusteddefenseproducts.org</code>                                        |
  | <code>query: freeman, bidget c., m.d \| 1805 center road town of saukville wi</code> | <code>query: freeman, bidgget c., m.d \| 1805 center road town of saukville wi</code> |
  | <code>query: z x & p digitalbridge, [llc] \| 877 terrace avenue chama nm</code>      | <code>query: z x & p [llc] digitalbridge, \| terrace avenue chama nm</code>           |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 128,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```
</details>
<details><summary>matches_US</summary>

#### matches_US

* Dataset: matches_US
* Size: 1,000,000 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                          | negative                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            | string                                                                            |
  | details | <ul><li>min: 11 tokens</li><li>mean: 16.07 tokens</li><li>max: 32 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 16.83 tokens</li><li>max: 33 tokens</li></ul> | <ul><li>min: 5 tokens</li><li>mean: 15.09 tokens</li><li>max: 36 tokens</li></ul> |
* Samples:
  | anchor                                                                                  | positive                                                                                 | negative                                                                                       |
  |:----------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
  | <code>query: select trading laboratories \| richton park il 5202 northwind drive</code> | <code>query: select trading laboratories inc \| 5202 northwind dr richton park il</code> | <code>query: northwind llc trading</code>                                                      |
  | <code>query: dykes safeguard, llc \| 3060 tar landing road shallotte nc</code>          | <code>query: dykes sfeaucrd, (llc) \| shallottetownship 3060 tar landing road nc</code>  | <code>query: national charities ltd \| 174 chadwick landing drive shallotte township nc</code> |
  | <code>query: ricci & flores inc \| 5411 kemosabe drive killeen tx</code>                | <code>query: ricci and flores inc \| 5411 kemosabe drive killeen tx</code>               | <code>query: ricci & flores midtown [(inc)] \| 5415 kemosabe drive killeen texas</code>        |
* Loss: [<code>CachedMultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "mini_batch_size": 128,
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```
</details>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 512
- `learning_rate`: 0.0001
- `num_train_epochs`: 1.0
- `warmup_steps`: 400
- `fp16`: True
- `dataloader_drop_last`: True
- `disable_tqdm`: True
- `ddp_timeout`: 600
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 512
- `per_device_eval_batch_size`: 8
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 0.0001
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 1.0
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: None
- `warmup_steps`: 400
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `enable_jit_checkpoint`: False
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `use_cpu`: False
- `seed`: 42
- `data_seed`: None
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: -1
- `ddp_backend`: None
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `disable_tqdm`: True
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `auto_find_batch_size`: False
- `full_determinism`: False
- `ddp_timeout`: 600
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `use_cache`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0062 | 50   | 1.4291        |
| 0.0125 | 100  | 0.6533        |
| 0.0187 | 150  | 0.3603        |
| 0.0250 | 200  | 0.2637        |
| 0.0312 | 250  | 0.1947        |
| 0.0375 | 300  | 0.1622        |
| 0.0437 | 350  | 0.1397        |
| 0.0500 | 400  | 0.1143        |
| 0.0562 | 450  | 0.0967        |
| 0.0625 | 500  | 0.0989        |
| 0.0687 | 550  | 0.0887        |
| 0.0750 | 600  | 0.0874        |
| 0.0812 | 650  | 0.0802        |
| 0.0875 | 700  | 0.0739        |
| 0.0937 | 750  | 0.0697        |
| 0.1000 | 800  | 0.0703        |
| 0.1062 | 850  | 0.0603        |
| 0.1124 | 900  | 0.0600        |
| 0.1187 | 950  | 0.0573        |
| 0.1249 | 1000 | 0.0503        |
| 0.1312 | 1050 | 0.0523        |
| 0.1374 | 1100 | 0.0551        |
| 0.1437 | 1150 | 0.0505        |
| 0.1499 | 1200 | 0.0475        |
| 0.1562 | 1250 | 0.0478        |
| 0.1624 | 1300 | 0.0461        |
| 0.1687 | 1350 | 0.0463        |
| 0.1749 | 1400 | 0.0457        |
| 0.1812 | 1450 | 0.0417        |
| 0.1874 | 1500 | 0.0395        |
| 0.1937 | 1550 | 0.0374        |
| 0.1999 | 1600 | 0.0399        |
| 0.2061 | 1650 | 0.0361        |
| 0.2124 | 1700 | 0.0363        |
| 0.2186 | 1750 | 0.0417        |
| 0.2249 | 1800 | 0.0365        |
| 0.2311 | 1850 | 0.0371        |
| 0.2374 | 1900 | 0.0347        |
| 0.2436 | 1950 | 0.0394        |
| 0.2499 | 2000 | 0.0385        |
| 0.2561 | 2050 | 0.0341        |
| 0.2624 | 2100 | 0.0296        |
| 0.2686 | 2150 | 0.0361        |
| 0.2749 | 2200 | 0.0323        |
| 0.2811 | 2250 | 0.0375        |
| 0.2874 | 2300 | 0.0280        |
| 0.2936 | 2350 | 0.0312        |
| 0.2999 | 2400 | 0.0283        |
| 0.3061 | 2450 | 0.0322        |
| 0.3123 | 2500 | 0.0347        |
| 0.3186 | 2550 | 0.0348        |
| 0.3248 | 2600 | 0.0321        |
| 0.3311 | 2650 | 0.0310        |
| 0.3373 | 2700 | 0.0305        |
| 0.3436 | 2750 | 0.0333        |
| 0.3498 | 2800 | 0.0301        |
| 0.3561 | 2850 | 0.0294        |
| 0.3623 | 2900 | 0.0282        |
| 0.3686 | 2950 | 0.0288        |
| 0.3748 | 3000 | 0.0290        |
| 0.3811 | 3050 | 0.0303        |
| 0.3873 | 3100 | 0.0285        |
| 0.3936 | 3150 | 0.0289        |
| 0.3998 | 3200 | 0.0264        |
| 0.4060 | 3250 | 0.0286        |
| 0.4123 | 3300 | 0.0250        |
| 0.4185 | 3350 | 0.0289        |
| 0.4248 | 3400 | 0.0218        |
| 0.4310 | 3450 | 0.0282        |
| 0.4373 | 3500 | 0.0279        |
| 0.4435 | 3550 | 0.0289        |
| 0.4498 | 3600 | 0.0269        |
| 0.4560 | 3650 | 0.0253        |
| 0.4623 | 3700 | 0.0282        |
| 0.4685 | 3750 | 0.0274        |
| 0.4748 | 3800 | 0.0248        |
| 0.4810 | 3850 | 0.0284        |
| 0.4873 | 3900 | 0.0262        |
| 0.4935 | 3950 | 0.0251        |
| 0.4998 | 4000 | 0.0253        |
| 0.5060 | 4050 | 0.0256        |
| 0.5122 | 4100 | 0.0231        |
| 0.5185 | 4150 | 0.0257        |
| 0.5247 | 4200 | 0.0234        |
| 0.5310 | 4250 | 0.0233        |
| 0.5372 | 4300 | 0.0289        |
| 0.5435 | 4350 | 0.0252        |
| 0.5497 | 4400 | 0.0232        |
| 0.5560 | 4450 | 0.0264        |
| 0.5622 | 4500 | 0.0260        |
| 0.5685 | 4550 | 0.0272        |
| 0.5747 | 4600 | 0.0256        |
| 0.5810 | 4650 | 0.0228        |
| 0.5872 | 4700 | 0.0257        |
| 0.5935 | 4750 | 0.0229        |
| 0.5997 | 4800 | 0.0201        |
| 0.6059 | 4850 | 0.0215        |
| 0.6122 | 4900 | 0.0202        |
| 0.6184 | 4950 | 0.0252        |
| 0.6247 | 5000 | 0.0258        |
| 0.6309 | 5050 | 0.0222        |
| 0.6372 | 5100 | 0.0236        |
| 0.6434 | 5150 | 0.0239        |
| 0.6497 | 5200 | 0.0244        |
| 0.6559 | 5250 | 0.0233        |
| 0.6622 | 5300 | 0.0240        |
| 0.6684 | 5350 | 0.0230        |
| 0.6747 | 5400 | 0.0252        |
| 0.6809 | 5450 | 0.0222        |
| 0.6872 | 5500 | 0.0221        |
| 0.6934 | 5550 | 0.0238        |
| 0.6997 | 5600 | 0.0235        |
| 0.7059 | 5650 | 0.0215        |
| 0.7121 | 5700 | 0.0218        |
| 0.7184 | 5750 | 0.0221        |
| 0.7246 | 5800 | 0.0222        |
| 0.7309 | 5850 | 0.0202        |
| 0.7371 | 5900 | 0.0224        |
| 0.7434 | 5950 | 0.0192        |
| 0.7496 | 6000 | 0.0236        |
| 0.7559 | 6050 | 0.0238        |
| 0.7621 | 6100 | 0.0194        |
| 0.7684 | 6150 | 0.0227        |
| 0.7746 | 6200 | 0.0193        |
| 0.7809 | 6250 | 0.0220        |
| 0.7871 | 6300 | 0.0217        |
| 0.7934 | 6350 | 0.0222        |
| 0.7996 | 6400 | 0.0239        |
| 0.8058 | 6450 | 0.0208        |
| 0.8121 | 6500 | 0.0212        |
| 0.8183 | 6550 | 0.0209        |
| 0.8246 | 6600 | 0.0233        |
| 0.8308 | 6650 | 0.0229        |
| 0.8371 | 6700 | 0.0197        |
| 0.8433 | 6750 | 0.0203        |
| 0.8496 | 6800 | 0.0188        |
| 0.8558 | 6850 | 0.0156        |
| 0.8621 | 6900 | 0.0206        |
| 0.8683 | 6950 | 0.0211        |
| 0.8746 | 7000 | 0.0185        |
| 0.8808 | 7050 | 0.0195        |
| 0.8871 | 7100 | 0.0219        |
| 0.8933 | 7150 | 0.0194        |
| 0.8996 | 7200 | 0.0184        |
| 0.9058 | 7250 | 0.0196        |
| 0.9120 | 7300 | 0.0189        |
| 0.9183 | 7350 | 0.0192        |
| 0.9245 | 7400 | 0.0211        |
| 0.9308 | 7450 | 0.0228        |
| 0.9370 | 7500 | 0.0182        |
| 0.9433 | 7550 | 0.0159        |
| 0.9495 | 7600 | 0.0192        |
| 0.9558 | 7650 | 0.0202        |
| 0.9620 | 7700 | 0.0197        |
| 0.9683 | 7750 | 0.0180        |
| 0.9745 | 7800 | 0.0220        |
| 0.9808 | 7850 | 0.0191        |
| 0.9870 | 7900 | 0.0200        |
| 0.9933 | 7950 | 0.0172        |
| 0.9995 | 8000 | 0.0190        |

</details>

### Training Time
- **Training**: 1.5 hours

### Framework Versions
- Python: 3.12.13
- Sentence Transformers: 5.4.1
- Transformers: 5.0.0
- PyTorch: 2.10.0+cu128
- Accelerate: 1.13.0
- Datasets: 5.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### CachedMultipleNegativesRankingLoss
```bibtex
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->