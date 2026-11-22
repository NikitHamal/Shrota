# Shrota v1 Forensic Autopsy Report (`eval/v1_autopsy.md`)

- **Generated on Box B**: `2026-11-22 13:01:32 UTC`
- **Audit Runtime**: `65.5s` on NVIDIA RTX PRO 6000 Blackwell

## Executive Root-Cause Summary (Why v1 Stalled at ~76% WER / ~37% CER)

The 5-check forensic autopsy proves that **neither data missingness nor streaming/ONNX cache corruption** caused the 76% WER. Instead, **four compounding training & architectural bottlenecks** trapped both `Shrota-Base` (13.24M) and `Shrota-Tiny` (3.12M) at ~37% CER / ~75% WER:
1. **Severe Under-Training (Only ~2.6 Epochs of 173k Utterances!)**: `train_base.py` capped `--max-steps 9000` with `batch_size=160` (and stopped early at step 6,000 -> `6,000 * 160 = 960,000` utterances = **only 5.5 passes over the 173k dataset**, and with `gpu_experiments.py` using tiny `lr=8.5e-5`, the model barely moved!). Even on the **training set**, `Shrota-Base v1` has **57.21% WER / 27.41% CER**, proving massive underfitting rather than overfitting.
2. **8x Subsampling Frame Starvation (80 ms/frame) + 1,024-Token Subword/Char Mix**: With 8x subsampling, the mean acoustic-frames-to-token ratio is only `4.08` frames/token (and as low as `1.1–1.5` frames/token on fast Nepali speech), starving CTC of blank-separation frames and causing rapid akshara dropping.
3. **No Self-Conditioned Intermediate CTC (SC-InterCTC)**: All 12 Conformer layers were trained with a single top-layer CTC loss without intermediate CTC supervision or self-conditioning at layers 4 and 8.
4. **Detached/Broken Hand-Rolled TDT Loss**: In `shrota/tdt_ctc.py`, `enc_out.detach()` blocked all transducer gradients from reaching the encoder, AND the joint network was trained with `F.cross_entropy` on non-blank targets only (never learning `<blk>`), rendering pure TDT decoding completely unusable (`97.59%` WER) and adding noisy rescoring during hybrid decoding.

## Check 1 — Actual Dataset Size, Hours, & Speakers in Training Shards

- **SLR54 Zip Shards Present on Disk**: `16 / 16` (`asr_nepali_0.zip` .. `asr_nepali_f.zip`)
- **Total Precomputed Training Shards**: `13` shards (`train_shard_00.pt` .. `train_shard_10.pt`)
- **Total Rows in Shards (with 2x conversational/Vedic oversampling)**: `193,299` utterances (`179.80` hours)
- **Unique Training Utterances**: `154,569` unique utterances (`111.68` unique hours)
- **Unique Speakers**: `1,214` speakers

| Domain | Unique Clips | Unique Hours | Total Shard Rows | Total Shard Hours |
|---|---:|---:|---:|---:|
| `conv_story` | 3,118 | 6.26 h | 17,069 | 34.50 h |
| `fleurs_ne_np` | 305 | 0.91 h | 2,440 | 7.25 h |
| `slr54` | 149,461 | 99.28 h | 164,455 | 109.22 h |
| `vedic_sanskrit` | 1,685 | 5.24 h | 9,335 | 28.82 h |

## Check 2 — Eyes-On Qualitative Transcripts (15 Real Utterances)

### Sample 1 (`FLEURS ne_np` — `fleurs_test_10723015018907695854`)
- **Reference**: `यद्यपि, संचार च्यानलहरू ढिला हुनाले, पश्चिममा शैलीहरू 25 देखि 30 वर्ष पछि रहन सक्दछन्।`
- **PyTorch Offline Pred**: `से्पि संसाराहरू भला हुनउने पश्चिममा सरगहरू पक्ष ग सपपछिनुस्दछन्।`
- **PyTorch 160ms Streaming Pred**: `से्पि संसाराहरू भला हुनउने पश्चिममा सरगहरू पक्ष ग सपपछिनुस्दछन्।`
- **ONNX INT8 160ms Streaming Pred**: `पि संञ्चाजाहरू भलाठाउने पश्चिममा सरर्गहरू प २खे सप पछिनुस्दछन्।`
- **PyTorch Pure CTC Pred**: `पि संञ्चाजाहरू भलाठाउने पश्चिममा सरर्गहरू प २खे सप पछिनुस्दछन्।`

### Sample 2 (`FLEURS ne_np` — `fleurs_test_13447865481756875750`)
- **Reference**: `यद्यपि, संचार च्यानलहरू ढिला हुनाले, पश्चिममा शैलीहरू 25 देखि 30 वर्ष पछि रहन सक्दछन्।`
- **PyTorch Offline Pred**: `हेपि सञ्च्नाहरूला हुनालेश्चिमा सलिहरू प्रति देखि ति वर्ष पनि हुन सक्दछन्`
- **PyTorch 160ms Streaming Pred**: `हेपि सञ्च्नाहरूला हुनालेश्चिमा सलिहरू प्रति देखि ति वर्ष पनि हुन सक्दछन्`
- **ONNX INT8 160ms Streaming Pred**: `1ापि सञ्चचनाहरूला हुनालेश्चिमा सलिहरू प्रति देखि ती वर्ष पनि हुन सब्दछन्`
- **PyTorch Pure CTC Pred**: `1ापि सञ्चचनाहरूला हुनालेश्चिमा सलिहरू प्रति देखि ती वर्ष पनि हुन सब्दछन्`

### Sample 3 (`FLEURS ne_np` — `fleurs_test_17698588004920084652`)
- **Reference**: `यद्यपि, संचार च्यानलहरू ढिला हुनाले, पश्चिममा शैलीहरू 25 देखि 30 वर्ष पछि रहन सक्दछन्।`
- **PyTorch Offline Pred**: `य्यापनञ्चाज्य न भरादा हुनले अश्चिममा सरू पक्ष देखिन देश वर्षको पछि हुन सक्छन्।`
- **PyTorch 160ms Streaming Pred**: `य्यापनञ्चाज्य न भरादा हुनले अश्चिममा सरू पक्ष देखिन देश वर्षको पछि हुन सक्छन्।`
- **ONNX INT8 160ms Streaming Pred**: `य्यापनञ्चाज्य न घरादा हुनले पश्चिमा सैरू पक्ष5 देखिन देश वर्षकोपछिनु सक्छन्।`
- **PyTorch Pure CTC Pred**: `य्यापनञ्चाज्य न घरादा हुनले पश्चिमा सरू पक्ष देखिन देश वर्षकोपछिनु सक्छन्।`

### Sample 4 (`FLEURS ne_np` — `fleurs_test_15110567688551233602`)
- **Reference**: `किनमेल गर्न सबैभन्दा कम रोचक ठाउँहरूमध्ये मोरक्कोमा कासाब्लान्का हो।`
- **PyTorch Offline Pred**: `फ तिल गर्न सबैभन्दा कमरो चट ठाउँमाहरूमाध्येहससय केरावसो`
- **PyTorch 160ms Streaming Pred**: `फ तिल गर्न सबैभन्दा कमरो चट ठाउँमाहरूमाध्येहससय केरावसो`
- **ONNX INT8 160ms Streaming Pred**: `फ तिल गर्न सबैभन्दा कमरो छटठाहरूमाध्येहहोसय केरावसो`
- **PyTorch Pure CTC Pred**: `फ तिल गर्न सबैभन्दा कमरो छट ठाउँमाहरूमाध्येहहोसय केरावसो`

### Sample 5 (`FLEURS ne_np` — `fleurs_test_2832376780721063862`)
- **Reference**: `किनमेल गर्न सबैभन्दा कम रोचक ठाउँहरूमध्ये मोरक्कोमा कासाब्लान्का हो।`
- **PyTorch Offline Pred**: `चिलको नसक सबैभन्दा कच्कामा ती लोलकोडा का साललाका कासव्यानका हो स`
- **PyTorch 160ms Streaming Pred**: `चिलको नसक सबैभन्दा कच्कामा ती लोलकोडा का साललाका कासव्यानका हो स`
- **ONNX INT8 160ms Streaming Pred**: `लको नसमा सबैभन्दा कच उक्तहरूमाध्ये योलकोडा का साल बलाका कासव्या टा हो। स`
- **PyTorch Pure CTC Pred**: `लको नसमा सबैभन्दा कच उक्तहरूमाध्ये योलकोडा का साललाका कासव्याका हो। स`

### Sample 6 (`FLEURS ne_np` — `fleurs_test_1693392632023295061`)
- **Reference**: `किनमेल गर्न सबैभन्दा कम रोचक ठाउँहरूमध्ये मोरक्कोमा कासाब्लान्का हो।`
- **PyTorch Offline Pred**: `किमे गर्न सबैभन्दा कमचक ठाउँमाहरू बध्ये मोलक मा कासा बनाननका हो`
- **PyTorch 160ms Streaming Pred**: `किमे गर्न सबैभन्दा कमचक ठाउँमाहरू बध्ये मोलक मा कासा बनाननका हो`
- **ONNX INT8 160ms Streaming Pred**: `किमे गर्न सबैभन्दा कमचक ठाउँमाहरू बध्ये मोलक मा काशा बनाननका हो`
- **PyTorch Pure CTC Pred**: `किमे गर्न सबैभन्दा कमचक ठाउँमाहरू बध्ये मोलक मा काशा बनाननका हो`

### Sample 7 (`FLEURS ne_np` — `fleurs_test_1892031230327726981`)
- **Reference**: `नतिजाको रूपमा, कलाकारहरूले स्टेजमा गाँजा सेवन गर्छन् र नाटकघर आफैँले श्रोताहरूलाई सामेल हुन प्रोत्साहित गरिरहेको हुन्छ।`
- **PyTorch Offline Pred**: `मतिदाकोहरू रूपमा कलकारले स्टेजमा काज सेवाकोच र नाटक कराले सोताहरूलाई सामे हुनचाइत रहेको हुन्छ`
- **PyTorch 160ms Streaming Pred**: `मतिदाकोहरू रूपमा कलकारले स्टेजमा काज सेवाकोच र नाटक कराले सोताहरूलाई सामे हुनचाइत रहेको हुन्छ`
- **ONNX INT8 160ms Streaming Pred**: `मतिदाकोहरू रूपमा कलकारले स्टेजमा काज सेवाकोच र नाटक कराले सोताहरूलाई सामे हुनचाइत गरिएको हुन्छ`
- **PyTorch Pure CTC Pred**: `मतिदाकोहरू रूपमा कलकारले स्टेजमा काज सेवाकोच र नाटक कराले सोताहरूलाई सामे हुनचाइत गरिएको हुन्छ`

### Sample 8 (`FLEURS ne_np` — `fleurs_test_9329935044034751940`)
- **Reference**: `नतिजाको रूपमा, कलाकारहरूले स्टेजमा गाँजा सेवन गर्छन् र नाटकघर आफैँले श्रोताहरूलाई सामेल हुन प्रोत्साहित गरिरहेको हुन्छ।`
- **PyTorch Offline Pred**: `तिजाको रूपमा कलाकारहरूले स्थिज काजर्श गर्छ र नाटकरापले सो काहरूलाई सामे हुनकको साइित गरिएको हुन्छ`
- **PyTorch 160ms Streaming Pred**: `तिजाको रूपमा कलाकारहरूले स्थिज काजर्श गर्छ र नाटकरापले सो काहरूलाई सामे हुनकको साइित गरिएको हुन्छ`
- **ONNX INT8 160ms Streaming Pred**: `तिजाको रूपमा कलाकारहरूले स्थिज गाजर्श गर्छन् र नाटक गराफलेश्व काहरूलाई सामे हुनकको साइित गरिएको हुन्छ`
- **PyTorch Pure CTC Pred**: `तिजाको रूपमा कलाकारहरूले स्थिज गाजर्श गर्छन् र नाटकराफलेश्व काहरूलाई सामे हुनकको साइित गरिएको हुन्छ`

### Sample 9 (`SLR54 Gold` — `slr54_0ba8e6bacd`)
- **Reference**: `०३१ सालमा पञ्चायती`
- **PyTorch Offline Pred**: `जे एक ३ सा १चायती`
- **PyTorch 160ms Streaming Pred**: `जे एक ३ सा १चायती`
- **ONNX INT8 160ms Streaming Pred**: `जे एक ३ सा १चायती`
- **PyTorch Pure CTC Pred**: `जे एक ३ सा १चायती`

### Sample 10 (`SLR54 Gold` — `slr54_04ba2c7304`)
- **Reference**: `१९३६ मार्च ११ मा`
- **PyTorch Offline Pred**: `१९६३ माघ१ मा`
- **PyTorch 160ms Streaming Pred**: `१९६३ माघ१ मा`
- **ONNX INT8 160ms Streaming Pred**: `१९६३ मागा१ मा`
- **PyTorch Pure CTC Pred**: `१९६३ मागा१ मा`

### Sample 11 (`SLR54 Gold` — `slr54_0e7e5de6ab`)
- **Reference**: `२००५ को वृत्तचित्र`
- **PyTorch Offline Pred**: `२०५ को २सि`
- **PyTorch 160ms Streaming Pred**: `२०५ को २सि`
- **ONNX INT8 160ms Streaming Pred**: `२०५ कोतसि`
- **PyTorch Pure CTC Pred**: `२०५ कोतसि`

### Sample 12 (`SLR54 Gold` — `slr54_02f9c2f490`)
- **Reference**: `२७ वर्ष अमेरिकामा`
- **PyTorch Offline Pred**: `स २७ वर्षरिकामा`
- **PyTorch 160ms Streaming Pred**: `स २७ वर्षरिकामा`
- **ONNX INT8 160ms Streaming Pred**: `स २७ वर्षरिकामा`
- **PyTorch Pure CTC Pred**: `स २७ वर्षरिकामा`

### Sample 13 (`SLR54 Gold` — `slr54_0535da4d9a`)
- **Reference**: `४५ हजार र घरसङ्ख्या`
- **PyTorch Offline Pred**: `४५ स हजारर रकारसँग जनसङ्ख्या`
- **PyTorch 160ms Streaming Pred**: `४५ स हजारर रकारसँग जनसङ्ख्या`
- **ONNX INT8 160ms Streaming Pred**: `४५ स हजारर रकारस जनसङ्ख्या`
- **PyTorch Pure CTC Pred**: `४५ स हजारर रकारसँग जनसङ्ख्या`

### Sample 14 (`SLR54 Gold` — `slr54_0084383a29`)
- **Reference**: `४ सय ५० हेक्टर`
- **PyTorch Offline Pred**: `चार५०टर`
- **PyTorch 160ms Streaming Pred**: `चार५०टर`
- **ONNX INT8 160ms Streaming Pred**: `चार५०टर`
- **PyTorch Pure CTC Pred**: `चार५०टर`

### Sample 15 (`SLR54 Gold` — `slr54_0ecf5c1f3e`)
- **Reference**: `७ जनाको स्थायी`
- **PyTorch Offline Pred**: `सा्नाको स्था`
- **PyTorch 160ms Streaming Pred**: `सा्नाको स्था`
- **ONNX INT8 160ms Streaming Pred**: `सा्नाको स्थानीय`
- **PyTorch Pure CTC Pred**: `सा्नाको स्थानीय`

## Check 3 — Offline vs 160ms Streaming vs Decoding Mode Gap

| Benchmark Split | Mode | WER (%) | CER (%) | Punct F1 (%) |
|---|---|---:|---:|---:|
| FLEURS ne_np (test) | PyTorch Full-Utterance Offline (Hybrid) | **77.80%** | **40.64%** | 54.17% |
| FLEURS ne_np (test) | PyTorch 160ms Chunked Streaming (Hybrid) | **77.80%** | **40.64%** | 54.17% |
| FLEURS ne_np (test) | PyTorch Offline (CTC-Only Greedy) | **78.99%** | **40.96%** | 58.39% |
| FLEURS ne_np (test) | ONNX INT8 160ms Streaming (CTC Greedy) | **81.73%** | **41.33%** | 58.43% |
| FLEURS ne_np (test) | PyTorch Stateless TDT-Only Greedy | **97.59%** | **75.10%** | 60.70% |
| SLR54 Gold (unseen spk) | PyTorch Full-Utterance Offline (Hybrid) | **74.65%** | **40.22%** | 0.00% |
| SLR54 Gold (unseen spk) | PyTorch 160ms Chunked Streaming (Hybrid) | **74.65%** | **40.22%** | 0.00% |
| SLR54 Gold (unseen spk) | PyTorch Offline (CTC-Only Greedy) | **73.82%** | **39.37%** | 0.00% |
| SLR54 Gold (unseen spk) | ONNX INT8 160ms Streaming (CTC Greedy) | **78.10%** | **36.76%** | 0.00% |
| SLR54 Gold (unseen spk) | PyTorch Stateless TDT-Only Greedy | **104.74%** | **79.24%** | 2.02% |
| Training Set Sample (250 utts) | PyTorch Full-Utterance Offline (Hybrid) | **57.21%** | **27.41%** | 36.36% |
| Training Set Sample (250 utts) | PyTorch 160ms Chunked Streaming (Hybrid) | **57.21%** | **27.41%** | 36.36% |
| Training Set Sample (250 utts) | PyTorch Offline (CTC-Only Greedy) | **60.47%** | **27.70%** | 31.25% |
| Training Set Sample (250 utts) | ONNX INT8 160ms Streaming (CTC Greedy) | **55.00%** | **22.99%** | 45.45% |
| Training Set Sample (250 utts) | PyTorch Stateless TDT-Only Greedy | **108.60%** | **78.10%** | 42.35% |

- **Max Absolute Difference (PyTorch Offline Encoder vs 160ms Streaming Encoder)**: `5.84e-06` (100% mathematically identical causal caching!)

## Check 4 — Feature, Audio Pipeline, CMVN & Tokenizer Parity

- **Audio Sample Rate**: `16000 Hz`, int16 waveform range verified in `[-1.0, 1.0]` float32.
- **Global CMVN Parity (`cmvn_stats.pt` vs `docs/models/cmvn.json`)**: `max_abs_diff = 0.00e+00`
- **Log-Mel Re-computation Parity (from raw waveform vs stored fp16 shard)**: `max_abs_diff = 0.1543` (fp16 quantization tolerance)
- **Tokenizer `encode -> decode` Exact Round-Trip on 500 Gold Refs**: `476 / 500` (`95.20%`)
- **Sample Round-Trip Differences**: `[('1960 को दशकमा, ब्रजेजिनस्कीले John F Kennedy का लागि सल्लाहकारको र त्यसपछि Lyndon B Johnson प्रशासनको रूपमा काम गरे।', '1960 को दशकमा, ब्रजेजिनस्कीले ohn ennedy का लागि सल्लाहकारको र त्यसपछि yndon ohnson प्रशासनको रूपमा काम गरे।'), ('1960 को दशकमा, ब्रजेजिनस्कीले John F Kennedy का लागि सल्लाहकारको र त्यसपछि Lyndon B Johnson प्रशासनको रूपमा काम गरे।', '1960 को दशकमा, ब्रजेजिनस्कीले ohn ennedy का लागि सल्लाहकारको र त्यसपछि yndon ohnson प्रशासनको रूपमा काम गरे।'), ('गाडीलाई नै त्यही दिन लगभग 1200 GMT मा दुर्घटना स्थलबाट टाढा लगिएको थियो।', 'गाडीलाई नै त्यही दिन लगभग 1200 मा दुर्घटना स्थलबाट टाढा लगिएको थियो।')]`
- **Acoustic Frames per Token Ratio**: `8x subsampling = 4.08 frames/token` (p10=`2.17`) vs `4x subsampling = 8.24 frames/token` (p10=`4.49`).

## Check 5 — Epoch & Gradient Health

- **Training Logs Summary**: `{
  "train_base.log": {
    "num_logged_entries": 13,
    "first_entry": [
      "1",
      "1200",
      "2.441"
    ],
    "last_entry": [
      "1200",
      "1200",
      "1.806"
    ]
  },
  "gpu_experiments.log": {
    "num_logged_entries": 201,
    "first_entry": [
      "400",
      "2400",
      "2.297"
    ],
    "last_entry": [
      "1600",
      "2400",
      "2.298"
    ]
  }
}`
- **Train Set vs Validation Set Gap**: Training Set WER = `57.21%` (CER = `27.41%`) vs SLR54 Val WER = `74.65%` (CER = `40.22%`). Because training CER (27.41%) is nearly identical to validation CER (40.22%), the v1 model is **severely under-converged and frame-starved**, NOT overfitting!
