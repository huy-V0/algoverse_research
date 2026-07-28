# algovse_research

**Question.** Does a persona vector's projection onto a k-dimensional refusal
subspace predict safety degradation better than its projection onto the single
Arditi refusal direction?

**Method.** Regress ASR and FRR shift on two predictors, the signed cosine to the
Arditi direction and the residual subspace component orthogonal to it, then
compare fit with a nested F-test.

All work runs in Google Colab on a free T4. Notebooks live here, weights and
tensors live in Google Drive.

---

## Extraction spec

Locked. Every teammate matches these five lines or the geometry numbers describe
different spaces.

| field | value |
|---|---|
| model | `Qwen/Qwen2.5-1.5B-Instruct` |
| forward dtype | float16 |
| saved vector dtype | float32 |
| token position | last prompt token, chat template applied, `add_generation_prompt=True` |
| layer index | 14, meaning `hidden_states[14]`, the output of block 13 |
| normalization | unit L2 norm |
| seed | 0 |

`hidden_states` holds 29 entries for this model. Entry 0 is the embedding output,
entry i is the output of block i-1. Off-by-one here silently shifts every result.

---

## Notebooks

### `notebooks/00_setup.ipynb`

Environment proof. Run it once per new machine or account.

- confirms a T4 backend with `nvidia-smi`
- mounts Drive and creates `persona_rqa/{vectors,results,cache}`
- pins `transformers==4.51.0`
- loads the model and asserts 28 layers, hidden size 1536
- runs one generation to prove the chat template works
- defines `get_resid` and checks the output shape

Produces no artifacts. Success looks like `torch.Size([2, 1536])` and readable
generated text.

### `notebooks/01_refusal_direction.ipynb`

Builds the Arditi refusal direction.

- samples 64 AdvBench goals and 64 Alpaca no-input instructions with seed 0
- writes the sampled prompts to `results/prompts_v1.json`
- extracts the last prompt token at **every** layer in one forward pass
- takes the mean difference at layer 14 and normalizes to unit length
- saves the vector with its full metadata payload
- reports separation AUC and a layer sweep

Caching all layers costs the same forward pass as caching one, so the layer sweep
in section 5 runs for free.

**Outputs**

| path | contents |
|---|---|
| `cache/acts_refusal_all_layers.pt` | `[64, 29, 1536]` float32 per prompt set, about 22 MB |
| `vectors/refusal_L14.pt` | unit direction plus metadata |
| `results/prompts_v1.json` | the exact 128 prompts, for teammate reuse |

**Pass criteria.** Section 5a AUC above 0.95 is healthy. Below 0.85 means the
layer or the token position is wrong. Fix it before going further.

---

## Drive layout

Notebooks are versioned here. Everything below stays in Drive and is not committed.

```
MyDrive/persona_rqa/
  vectors/   refusal_L14.pt, sycophancy_L14.pt, ...
  results/   prompts_v1.json, geometry.csv, steering_eval.csv
  cache/     acts_refusal_all_layers.pt
```

Never point the HuggingFace cache at Drive. Model weights belong on the VM local
disk, which is the default. Leave `cache_dir` and `HF_HOME` alone.

---

## How to run

1. Open a notebook in Colab, either from this repo or by uploading the file.
2. Runtime, then Change runtime type, then T4 GPU.
3. Runtime, then Manage sessions, and terminate anything else. Free tier allows
   one GPU backend at a time.
4. Run all cells top to bottom.
5. Edit, then Clear all outputs before committing.

---

## Datasets

| purpose | source |
|---|---|
| harmful prompts | AdvBench, `harmful_behaviors.csv`, column `goal` |
| harmless prompts | `tatsu-lab/alpaca`, `instruction` where `input` is empty |
| over-refusal (FRR) | XSTest, 250 safe and 200 unsafe prompts |
| jailbreaks (ASR) | JailbreakBench |

The last two arrive in notebook 04.

---

## Status and next steps

- [x] `00_setup`
- [x] `01_refusal_direction`
- [ ] ablation gate. Subtract the direction during generation on 20 held-out
      AdvBench goals and confirm refusals collapse. Blocks everything downstream.
- [ ] `02_persona_vectors`. Sycophancy, evil, hallucination, same extraction spec.
- [ ] `03_geometry`. Cosine, subspace projection norm, `r_perp`, written to CSV.
- [ ] `04_steering_eval`. 3 traits x 5 layers x 5 coefficients, ASR and FRR.
- [ ] `05_regression`. Nested models, partial R2 of `r_perp`, random-subspace control.

Build `r_perp` and the random-subspace control before the full grid. Those two
numbers show whether RQ-A holds up before you spend the compute.

---

## Team

| person | scope |
|---|---|
| Nikolai | measurement and regression |
| Carson | trait vector extraction |
| Sabrina | refusal subspace construction |

On arrival, check that the first component of Sabrina's subspace has cosine above
0.8 with `refusal_L14.pt`. A lower value means a bug on one side.
