# NeuroGate Demos
<p align="center">
  <img src="https://github.com/gluedays-cyber/neurogate/raw/main/assets/neurogate-hero.jpg" width="100%" alt="NeuroGate vs Retro Branching — Electric Hyperbike vs Rusty Bicycle">
</p>
  <strong>Interactive On-The-Fly Training & Live Branching Showcases in Pure Go</strong><br>
  <em>Enterprise demonstration suite for <a href="https://github.com/gluedays-cyber/neurogate"><strong>NeuroGate</strong></a>. Creates domain artificial intelligence from scratch in ~1.5s, routing execution flow in ~30 μs with 0 B/op (Zero Allocations), Zero Downloads, and Zero CGO.</em>
</p>

<p align="center">
  <a href="#benchmarks"><img src="https://img.shields.io/badge/Latency-~30_μs-brightgreen.svg" alt="Latency"></a>
  <a href="#benchmarks"><img src="https://img.shields.io/badge/Allocs-0_B/op_(0_allocs)-blue.svg" alt="Allocations"></a>
  <img src="https://img.shields.io/badge/Wire_Format-v2_Positional-orange.svg" alt="Format v2">
  <img src="https://img.shields.io/badge/CGO-Zero_Disabled-success.svg" alt="CGO Zero">
  <img src="https://img.shields.io/badge/Go-1.21+-00ADD8.svg" alt="Go Version">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License">
</p>

<p align="center">
  <a href="docs/MANUAL.md"><strong>📖 Read the Full Developer Manual & Production Tutorial →</strong></a>
</p>

---

## What is NeuroGate?

**NeuroGate does NOT borrow, lease, or download external AI models. This engine directly creates and runs its own domain artificial intelligence from scratch.**

Instead of relying on brittle regex matching or calling bloated external LLMs, it **manufactures a domain-specific lightweight neural network directly from your dataset in under 2 seconds**. It maps typos, slang, inverted syntax, and colloquial phrasing into a continuous latent vector space—routing execution flow directly to your bound Go functions in **microseconds (~30 μs) with strictly 0 B/op heap allocation**.

```
Incoming Request ("bruh can u refund order #49281")
                     │
                     ▼
       [ In-Memory BPE Tokenizer ]
                     │
                     ▼
  [ Dense (D=64) + Positional (P=32x64) ]
                     │
                     ▼
  [ Non-Linear GELU Mean Pooling (D=64) ]
                     │
                     ▼
      [ Hidden Projection (D=128) ]
                     │
                     ▼
   [ Softmax + Shannon Entropy Calibrated Guard ]
                     │
     ┌───────────────┼───────────────┬────────────────┐
     ▼               ▼               ▼                ▼
(Score ≥ 0.75)  (Score ≥ 0.30)  (Margin < 0.15)  (Entropy > 2.0 / UNK ≥ 0.5)
[DEFINITE ROUTE] [PIPELINE]      [AMBIGUOUS]      [FALLBACK ISOLATION]
```

---

## Why NeuroGate? (Beyond Retro Branching, Cloud LLMs, and Bloated Local Models)

Modern backends face an architectural dilemma when routing unstructured or noisy user requests:

```go
// ❌ RETRO BRANCHING: Brittle, explodes in complexity, collapses under real-world noise
if strings.Contains(input, "refund") || strings.Contains(input, "cancel") {
    // FAILS on: "sent the return box a week ago when do i get my money back"
    // FAILS on: "can u reverse the charge?" (typos, slang, synonyms)
    // MISROUTES on: "cancel shipment delay notifications" (word collision)
}

// ❌ CLOUD LLMs: Massive network latency, recurring per-token cost, third-party dependency
// Latency: 400ms – 2,500ms (Unusable in high-throughput microservices)
// Cost: $0.0015 – $0.03 per request (Bills explode under scale)
// Vulnerability: Outages, rate limits, JSON hallucination, network partitions

// ❌ LOCAL LLMs & SLMs (Ollama, llama.cpp, Mistral-7B, Phi-3): Severe host resource exhaustion
// Memory: Monopolizes 4.5 GB to 8.0 GB+ of RAM/VRAM just to pick a 4-byte enum
// CPU Starvation: Burns 100% CPU across multiple cores, starving companion microservices
// Deployment Complexity: Requires CGO, C++ shared libraries (libllama.so), or background daemons

// ✅ NEUROGATE: Self-Generated Micro-AI (In-Memory Go Engine)
// Memory Footprint: Under 180 KB (25,000x smaller than quantized 7B models)
// Latency: ~30 μs with 0 B/op (0 allocs) and deterministic 3-tier fallback
// Deployment: 100% Pure Go with CGO_ENABLED=0 single static binary
```

### Architectural Comparison Matrix

| Capability | Retro Branching (`if` / Regex) | Cloud LLMs (OpenAI / Claude) | Local LLMs (Ollama / llama.cpp) | **NeuroGate v2.0 (Embedded Engine)** |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Latency** | < 1 μs | 300 ms – 2,500 ms (Network bound) | 30 ms – 300 ms (Compute bound) | **~30 μs (In-Memory)** |
| **Throughput (per core)** | > 500,000 req/sec | ~50 req/sec (Rate limited) | ~20–50 req/sec (CPU saturated) | **> 33,000 req/sec (Zero Alloc)** |
| **Runtime Allocation** | 0 B/op | High (HTTP payload) | High (CGO buffers) | **0 B/op (0 allocs/op)** |
| **System Memory (RAM)** | Negligible | External service | **4.5 GB – 8.0 GB+ (VRAM / RAM)** | **< 180 KB (Format v2)** |
| **Token Order Awareness** | Rigid regex position | ✅ Transformer Attention | ✅ Transformer Attention | ✅ **Learned Positional Embeddings** |
| **Hardware Reqs** | Standard CPU | External service | High-end GPU or 8+ Core CPU | **Runs on a $5 VPS (16MB container)** |
| **Operational Cost** | $0.00 | $0.0015+ per call | High hardware/electricity cost | **$0.00 (Self-contained)** |
| **Hot Weight Reload** | Binary recompile | API model string switch | Multi-second model reload | **Lock-free Atomic Hot-Swap (`0 ns` stop)** |
| **Active Learning Loop** | N/A | Manual logging | N/A | **Built-in Ring Buffer Telemetry** |
| **Deployment Complexity** | Single binary | API client | CGO / C++ runtime / Ollama daemon | **Pure Go (`CGO_ENABLED=0`)** |

---

## Key Highlights

- **Zero Downloads & On-The-Fly AI Creation**: You never download gigabytes of pre-trained weights from HuggingFace or lease external APIs. NeuroGate forges a domain neural AI model directly from your CSV in under 2 seconds.
- **Zero Allocations on Hot Path (`0 B/op`)**: `PredictSlots` executes inference without triggering GC pressure, returning zero-heap stack results.
- **Semantic XOR & Word Order Disambiguation**: Format v2 embeds 32 positional vectors coupled with non-linear $GELU(E_i + P_i)$ pooling, mathematically distinguishing permutations like `"delivery refund"` from `"refund delivery"`.
- **3-Tier Decision Pipeline**: Classifies predictions into **Definite** (High confidence), **Ambiguous** (Borderline/narrow margin), or **Fallback** (Out-of-Distribution / High Shannon Entropy).
- **Multi-Intent Pipeline Support**: Automatically executes composite pipelines when secondary intent confidence meets multi-intent thresholds.
- **Lock-Free Atomic Hot-Swap & Telemetry**: Replace model weights on live traffic without locks (`sync/atomic.Pointer`), and stream drift queries into a bounded ring buffer for active learning.

---

## Benchmarks

Benchmarked on an AMD Ryzen 5 5600H (12 threads) running pure Go standard runtime (`go test -bench="." -benchmem`):

| Benchmark Target | Ops / Sec | Latency | Memory / Op | Allocations |
| :--- | :--- | :--- | :--- | :--- |
| **`BenchmarkPredictSlots`** | **33,433 ops/sec** | **29.91 μs** | **0 B/op** | **0 allocs/op** |
| **`BenchmarkPredictTokens`** | **33,126 ops/sec** | **30.18 μs** | **0 B/op** | **0 allocs/op** |
| **`BenchmarkForward`** | **33,091 ops/sec** | **30.21 μs** | **24 B/op** | **1 allocs/op** |
| **`BenchmarkGELU`** | **494,071 ops/sec** | **2.02 μs** | **0 B/op** | **0 allocs/op** |

---

## Installation

```bash
go get github.com/gluedays-cyber/neurogate
```

---

## 3-Step Lifecycle

### Step 1: AI Design — Prepare Your Domain Knowledge (`train.csv`)
Create a clean two-column CSV containing natural user queries and corresponding target labels:

```csv
text,label
I want to cancel my payment and request a refund,Refund
Where is my package and delivery tracking,Delivery
Forgot my account password please reset,Account
sent the return box a week ago when do i get my money back,Refund
yo i typed the wrong apt number please update address,Delivery
locked out of my account after 3 tries help pls,Account
```

### Step 2: Build Your Own AI — Programmatic In-Memory Compilation
Train your domain vocabulary and neural weights directly from Go code without external CLI tools:

```go
package main

import (
	"log"

	"github.com/gluedays-cyber/neurogate"
)

func main() {
	// Load training samples from dataset
	samples, err := neurogate.LoadCSVDataset("data/train.csv")
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}

	// Configure hyperparameters
	cfg := neurogate.DefaultTrainConfig()
	cfg.Epochs = 50
	cfg.LearningRate = 0.005
	cfg.TargetVocabSize = 250

	// Compile Little-Endian neural model in ~1.5s
	model, err := neurogate.TrainModel(samples, cfg)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	// Persist binary weights
	if err := neurogate.SaveBinaryModel("weights/model.bin", model); err != nil {
		log.Fatalf("Failed to save model: %v", err)
	}
	log.Println("Model compiled successfully.")
}
```
*(Note: A standalone CLI compiler `ib-train` is provided in the companion [NeuroGate - demo](../NeuroGate%20-%20demo) project).*

### Step 3: AI-Powered Branching — High-Level Routing Engine
Initialize the router using the high-level `OpenOrTrain` API (or `Open` if weights already exist), bind domain handlers, and execute microsecond routing:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gluedays-cyber/neurogate"
)

// 1. Business Logic Handlers
func handleRefund(ctx context.Context, payload any) error {
	fmt.Printf("[ACTION: Refund]   Processing refund for: '%v'\n", payload)
	return nil
}

func handleDelivery(ctx context.Context, payload any) error {
	fmt.Printf("[ACTION: Delivery] Querying shipment tracking for: '%v'\n", payload)
	return nil
}

func handleAccount(ctx context.Context, payload any) error {
	fmt.Printf("[ACTION: Account]  Initiating account security for: '%v'\n", payload)
	return nil
}

func handleFallback(ctx context.Context, payload any) error {
	fmt.Printf("[FALLBACK: Safety] Isolated low-confidence request: '%v'\n", payload)
	return nil
}

func main() {
	modelPath := "weights/model.bin"
	dataPath := "data/train.csv"

	// 1. Zero-boilerplate library API: Train or load compiled weights
	router, err := neurogate.OpenOrTrain(dataPath, modelPath, 0.60)
	if err != nil {
		log.Fatalf("NeuroGate engine initialization failed: %v", err)
	}

	// 2. Bind intelligent branches
	router.
		Branch("Refund", handleRefund).
		Branch("Delivery", handleDelivery).
		Branch("Account", handleAccount).
		Fallback(handleFallback)

	// 3. Execute microsecond branch dispatch
	testQueries := []string{
		"I want to cancel my payment and request a refund",
		"When will my delivery package arrive",
		"Forgot my account password",
		"Please refund my purchase",
		"Track my shipment status",
		"Completely random gibberish noise 12345!@#$",
	}

	fmt.Println("=== NeuroGate Server Routing Started ===")
	ctx := context.Background()
	for _, query := range testQueries {
		if err := router.Dispatch(ctx, query, query); err != nil {
			log.Printf("Dispatch error: %v", err)
		}
	}
	fmt.Println("=== All queries dispatched in microseconds ===")
}
```

### Advanced Production Pattern: Multi-Intent Pipeline, 3-Tier Policy & Hot-Reload
For mission-critical production services requiring 3-tier calibration, composite intent pipelines, and concurrent lock-free model hot-swapping:

```go
// 1. Configure 3-Tier Policy and Active Learning Telemetry Buffer
router.SetPolicy(neurogate.DispatchPolicy{
	HighThreshold:     0.75,
	LowThreshold:      0.40,
	MarginCutoff:      0.15,
	MaxEntropy:        2.0,
	PipelineThreshold: 0.30,
}).EnableTelemetry(1024)

// 2. Bind composite multi-intent and ambiguous handlers
router.
	BindPipeline("Refund", "Delivery", func(ctx context.Context, p, s string, payload any) error {
		fmt.Printf("[PIPELINE: %s -> %s] Processing combined return & shipment: %v\n", p, s, payload)
		return nil
	}).
	Ambiguous(func(ctx context.Context, p, s string, payload any) error {
		fmt.Printf("[AMBIGUOUS: %s vs %s] Requesting user confirmation: %v\n", p, s, payload)
		return nil
	})

// 3. Execute multi-intent pipeline dispatch
_ = router.DispatchPipeline(ctx, "can u cancel order #49281 and update delivery?", "OrderPayload")

// 4. Lock-free atomic hot-reload on live traffic (0 ns stop-the-world)
_ = router.Reload("weights/intent.bin")

// 5. Drain active learning drift events
events := router.DrainTelemetry()
fmt.Printf("Harvested %d drift events for active learning retraining.\n", len(events))
```

---

## Fail-Safe Safe Rejection & Active Learning Retraining Guide

NeuroGate v1.1.0 does not force classification on unlearned, out-of-domain (OOD), or high-entropy queries. Instead, it enforces **Safe Rejection**:

```
[ Unlearned / OOD Query ]
           │
           ▼
[ Layer 1 / Layer 2 Guardrails ] ──(Triggered)──> [ Safe Rejection & Fallback Isolation ]
                                                            │
                                                            ▼
                                              [ Active Learning Buffer ]
                                                            │
                                                            ▼
                                              [ Human Review & Labeling ]
                                                            │
                                                            ▼
                                              [ Dataset CSV + Variants ]
                                                            │
                                                            ▼
                                              [ Re-Compile & 0ns Reload ] ──> Processed Confidently in ~30 μs!
```

### 1. Handling Rejected Queries & Post-Retraining Resolution

When a query is rejected (e.g. `ErrUnlearnedVocabulary`, `ErrOutOfDomain`, or isolated to `Fallback`), the engine safely isolates execution rather than triggering unauthorized actions. 

1. **Harvest Rejected Queries**: Retrieve isolated inputs via telemetry (`DrainTelemetry()`) or fallback logger.
2. **Assign Ground-Truth Label**: Human reviewers verify the true business intent (e.g. marking an unlearned query as `Refund`).
3. **Append to Dataset**: Add the labeled statement to `dataset.csv`.
4. **Recompile & Atomic Reload**: Re-train model in ~1.5s and call `router.Reload("weights/model.bin")` or `gate.SwapModel(newModel)`.
5. **Immediate Resolution**: Subsequent occurrences of this query and its linguistic variations will be routed with microsecond latency and high confidence without falling back.

### 2. Best Practice: Augment with Similar Variants & Typo Patterns

When adding an unlearned query to the dataset, **do not add only a single sentence**. Adding 3 to 5 similar phrasing variants and common typo mutations alongside the original query significantly improves model robustness:

| Strategy | Query to Add | Why It Matters |
| :--- | :--- | :--- |
| **Original Rejected Query** | `"charge reversed to card plz"` | Captures the exact novel phrasing rejected by Layer 1/2. |
| **Synonym & Lexical Variants** | `"reverse the card transaction"`, `"credit back my card"` | Expands the BPE subword dictionary with diverse domain tokens. |
| **Inverted Syntax Variants** | `"to my card please refund charge"`, `"my card needs reverse charge"` | Trains Positional Encoding ($P_i$) to handle arbitrary grammatical order. |
| **Colloquial / Typo Variants** | `"card charg revers"`, `"plz revers card"` | Prevents future single-character fallback cutoff on messy mobile typing. |

> **Why Variant Augmentation Works**: NeuroGate's in-memory BPE tokenizer constructs subwords based on frequency statistics. Adding lexical clusters ensures that subword fragments (e.g., `revers`, `charg`, `card`) become atomic vocabulary items, preventing Layer 1 single-character fragmentation and building a tight, convex manifold in the latent vector space.

### 3. Model Training & Retraining Tips

1. **Maintain Balanced Class Representation**:
   - Ensure each class has roughly equal sample counts (minimum 20–30 sentences per intent).
   - Extreme class imbalance (> 3:1 ratio) can tilt Softmax priors toward the majority class.
2. **Tune `TargetVocabSize` to Domain Scope**:
   - Set between `200` and `350` for standard business domain routing.
   - If set too low (< 120), text breaks into raw 1-character fragments and triggers Layer 1 rejection.
   - If set too high (> 500) on a small dataset, embedding weights become sparse and prone to over-fitting.
3. **Hyperparameter Recommendations**:
   - `Epochs`: `35` to `50` (leveraging built-in `Patience = 10` early stopping).
   - `LearningRate`: `0.003` to `0.005` (with Adam optimizer).
   - `BatchSize`: `16` to `32`.
4. **Combine with Neuro-Symbolic Anchors (`WithAnchor`)**:
   - For mission-critical keywords (e.g., `"refund"`, `"charge"`, `"card"`), register anchor weights (`1.3` to `2.0`). This guarantees instantaneous 1-cycle bitmask logit boosting even on edge-case phrasing.
5. **Zero-Downtime Atomic Hot-Swapping**:
   - Use `router.Reload(path)` or `gate.SwapModel(model)` to apply re-trained weights live on active traffic with `0 ns` stop-the-world overhead.

---

## Observability & Whitebox Debugging

Need to understand why a query routed to a specific branch or why it fell back? Use `Inspect`:

```go
trace := router.Inspect("can u cancel order #49281? i bought it by mistake")
```

```json
{
  "input_text": "can u cancel order #49281? i bought it by mistake",
  "token_ids": [4, 5, 8, 12, 45, 98],
  "subwords": ["can", "u", "cancel", "order", "#", "mistake"],
  "unknown_token_ratio": 0.0,
  "class_probabilities": {
    "Account": 0.0012,
    "Delivery": 0.0035,
    "Refund": 0.9953
  },
  "predicted_label": "Refund",
  "secondary_label": "Delivery",
  "confidence": 0.9953,
  "margin": 0.9918,
  "entropy": 0.0351,
  "threshold": 0.75,
  "is_ambiguous": false,
  "is_pipeline": false,
  "is_fallback": false,
  "latency_micros": 30
}
```

---

## NeuroGate: 3-Head Geometric Intelligent Filtering Engine

`NeuroGate` is an in-memory intelligent filtering gate that wraps a single shared neural backbone encoder with three orthogonal geometric and symbolic guard heads. It solves model overconfidence, Out-of-Domain (OOD) leakage, and slang ambiguity without spawning multiple fragmented networks or incurring heap allocations.

```
Incoming Request ("it is too dark in here please switch on lamps")
                               │
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │ Shared Neural Backbone (In-Memory BPE + Positional MLP) │ ──> z ∈ ℝ⁶⁴ (Unit Norm)
  └────────────────────────────┬───────────────────────────┘
                               │
 ┌─────────────────────────────┴─────────────────────────────┐
 │ Stack-Allocated 3-Head Geometric Gate (~5 to ~28 μs)      │
 │                                                           │
 │  [Head 1]: L2 Cosine Out-of-Domain (OOD) Guard            │
 │            DotProduct(z, C_domain) < MinCosine ?          │
 │            --> Immediate Fallback Isolation if OOD        │
 │                                                           │
 │  [Head 2]: 1-Cycle Bitwise Symbolic Anchor Soft-Bias      │
 │            if (Bitmask & Anchor_i) != 0                   │
 │            --> Logit_i += Weight * PopCount(Mask)         │
 │                                                           │
 │  [Head 3]: Calibrated Top-2 Margin & Shannon Entropy      │
 │            Stack Softmax over Adjusted Logits             │
 │            --> Definite / Pipeline / Ambiguous / Fallback │
 └─────────────────────────────┬─────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
 [Definite Action]     [Multi-Intent Pipeline] [Safe Fallback]
```

### Key Engineering Capabilities

1. **Strict Zero-Allocation Hot-Path (`0 B/op`, `0 allocs/op`)**:
   Internal inference operates on fixed stack buffers (`[16]float32` and `[64]float32`). Pre-tokenized inputs dispatched via `FilterTokens` execute in **~28 μs with strictly 0 B/op heap allocation**.
2. **Single Shared Backbone (No Error Cascading)**:
   Avoids training multiple fragmented networks. A single compact encoder extracts context, while downstream safety boundaries and semantic boosts are computed geometrically.
3. **Neuro-Symbolic Anchor Soft-Bias**:
   Replaces fragile `strings.Contains` hardcoded branching with additive logit bonuses. Subword token IDs map to 64-bit masks (`uint64`), executing anchor boosts in a single CPU cycle (`&` and `popcount`).
4. **Geometric L2 Cosine OOD Boundary**:
   Compares normalized query embeddings against the calibrated domain manifold center ($C_{\text{domain}}$) using fast dot products, isolating OOD queries (e.g. quantum physics queries sent to an e-commerce router).

### NeuroGate Production Example

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/gluedays-cyber/neurogate"
)

func main() {
	// 1. Initialize NeuroGate from binary model
	gate, err := neurogate.NewNeuroGate("weights/demo_iot.bin")
	if err != nil {
		log.Fatalf("NeuroGate init failed: %v", err)
	}

	// 2. Calibrate domain manifold centroid from training samples
	if samples, err := neurogate.LoadCSVDataset("data/demo_iot.csv"); err == nil {
		gate.CalibrateDomainCentroid(samples)
	}

	// 3. Bind route handlers with symbolic anchor soft-biases
	gate.Bind("LightControl", func(ctx context.Context, payload any) error {
		fmt.Println(">>> [GPIO 18 HIGH] Toggle Living Room Chandelier")
		return nil
	}).WithAnchor(1.8, "dark", "light", "lamps", "lamp", "switch", "lights")

	gate.Bind("ClimateControl", func(ctx context.Context, payload any) error {
		fmt.Println(">>> [MODBUS UART] Set Daikin HVAC Inverter Temperature")
		return nil
	}).WithAnchor(1.8, "cooling", "heat", "fan", "temp", "ac", "air")

	gate.Fallback(func(ctx context.Context, payload any) error {
		fmt.Println(">>> [FALLBACK] Isolated Out-of-Domain or Ambiguous Request")
		return nil
	})

	// 4. Dispatch with microsecond latency and zero allocations
	ctx := context.Background()
	_ = gate.Filter(ctx, "it is too dark in here please switch on lamps", nil)
}
```

---

## The Evolution of Control Flow: Why Retro Branching Fails & How NeuroGate Proves Its Architectural Superiority

Traditional programming languages force engineers into **discrete control flow** (`if`, `switch`, `hash map`, `regex`). These constructs were invented in the 1960s for deterministic, byte-exact hardware primitives. When applied to real-world strings, natural language, unstructured logs, or conversational commands, **they collapse entirely**.

NeuroGate transforms control flow from brittle discrete matching into **continuous geometric vector-space routing ($text \to action$) in ~30 μs**.

The included multi-task demonstration driver (`cmd/ib-demo`) directly pits NeuroGate against traditional programming primitives across 6 critical enterprise domains:

### 1. Structural Comparison: Retro Branching vs. NeuroGate

| Control Flow Primitive | Why It Breaks Down on Real-World Input | How NeuroGate Resolves It Permanently |
| :--- | :--- | :--- |
| **`switch` / `if (str == val)`** | **100% Failure on Variations**: A 1-character typo (`"refnd"`), colloquial phrasing (`"gimme my cash back"`), or extra whitespace causes silent fall-through. | **BPE Continuous Embedding**: Maps all semantic synonyms and misspelled subwords to contiguous vector coordinates in 64-D space. |
| **Hash Maps (`map[string]T`)** | **Exact-Key Blindness**: Cannot index semantic equivalence. Caching 10,000 phrasing variations requires 10,000 distinct hash keys, leading to memory bloat and constant cache misses. | **Semantic Coordinate Resolution**: Resolves infinite sentence variations into deterministic Go handlers in ~30 μs with zero external network overhead. |
| **Regular Expressions (`regex`)** | **Combinatorial Explosion & ReDoS**: Supporting synonyms requires nested lookaheads and permutations ($O(N!)$ rules), causing CPU exhaustion (ReDoS backtracking) and unmaintainable regex hell. | **Non-Linear GELU Tensor Layers**: Evaluates feature cross-products without regex backtracking, maintaining deterministic, capped compute times. |
| **String Search (`strings.Contains`)** | **Semantic XOR Failure**: Commutative addition collapses opposite meanings. Cannot distinguish `"refund my delivery"` from `"delivery instead of refund"`. Word collisions cause catastrophic misrouting. | **Learned Positional Embeddings**: Encodes token sequence coordinates ($P_{32 \times 64}$) into non-linear activations, mathematically distinguishing token order permutations. |
| **Binary Boolean Decisions** | **Forced Misclassification**: Discrete `if/else` forces ambiguous or out-of-distribution noise into whichever branch happens to have a loose wildcard match. | **3-Tier Calibrated Pipeline**: Quantifies Shannon Entropy to isolate OOD noise to Fallback, while detecting competitive top-2 margins to trigger Step-up 2FA/Ambiguous logic. |

---

### 2. 6 Enterprise Domains: Proven Architectural Superiority

The companion demonstration project proves these architectural advantages live across 6 isolated production models:

#### Domain 1: E-Commerce CS Gateway (Defeating the Semantic XOR Dilemma)
- **The Retro Collapse**: `if strings.Contains(msg, "refund") && strings.Contains(msg, "delivery")` collapses opposite business intents. Both `"refund delivery fee"` and `"delivery instead of refund"` trigger the same branch. Regex permutations explode exponentially.
- **The NeuroGate Victory**: Learned positional vectors ($P_i$) coupled with non-linear $GELU(E_i + P_i)$ pooling mathematically separate token permutations. Furthermore, `DispatchPipeline` automatically executes composite operations (e.g. Return Approved $\to$ Reshipment Initiated) when both primary and secondary confidences qualify.

#### Domain 2: Semantic LLM Gateway (Defeating Hash Map Key Misses & API Waste)
- **The Retro Collapse**: Caching natural language with `map[string]Handler` achieves a near 0% hit rate because users never type the exact same string twice. Consequently, backends route 100% of routine traffic to OpenAI/Claude, burning $0.02–$0.05 and 1,500ms per request.
- **The NeuroGate Victory**: Maps routine banking commands (`QueryBalance`, `TransferFunds`, `CardLock`) directly to in-memory Go handlers in **30 μs at $0.00 cost**. Out-of-Distribution (OOD) queries (e.g. `"explain quantum physics"`) are detected via high Shannon Entropy ($> 1.80$) and safely escalated to cloud LLMs.

#### Domain 3: High-Throughput SRE Log Triage (Defeating ReDoS & GC Pauses with 0 B/op)
- **The Retro Collapse**: Ingesting 100,000+ log lines/sec through complex regex engines burns 100% CPU due to catastrophic backtracking. String allocations trigger GC stop-the-world pauses, choking message brokers (Kafka, Vector).
- **The NeuroGate Victory**: Executes stack-allocated zero-heap inference (`PredictSlots`) with **strictly 0 B/op and 0 allocs/op**. Instantly routes critical P0 panics (OOMKilled) to autoscalers while shunting low-priority health probes without heap garbage.

#### Domain 4: Offline Edge IoT Control (Defeating Brittle Keyword Matching in <180KB RAM)
- **The Retro Collapse**: Hard-coded `switch(cmd)` fails when users speak naturally: `"it's freezing in here"` fails to trigger `"turn on heater"`. Running local 7B models requires 4GB+ RAM, impossible on 64MB embedded Linux boards.
- **The NeuroGate Victory**: Compiles into a single Little-Endian binary under 180 KB with zero external dependencies and zero CGO. Maps colloquial voice/text variants directly to hardware GPIO/UART actuators in single-digit microseconds.

#### Domain 5: Automated CI/CD Failure Triage (Defeating Fragile String Scrapers)
- **The Retro Collapse**: Compiler error messages change formatting across toolchains (Docker, Go, Gradle, Kubernetes). Hard-coded string pattern matching silently breaks, forcing DevOps engineers to manually triage build failures.
- **The NeuroGate Victory**: Ingests unstructured build error tails and generalizes statistical subwords to trigger deterministic self-healing actions: `AutoRetry` (transient network 504), `ScaleUp` (OOM kill status 137), or `NotifyAuthor` (code syntax error).

#### Domain 6: FinTech Transaction Memo Audit (Defeating Naive Blacklists with 3-Tier Safety)
- **The Retro Collapse**: Keyword blacklists (`strings.Contains("scam")`) are trivially bypassed by fraudsters using typo obfuscation (`"p0lice f1ne"`). Rigid binary `if/else` either blocks legitimate transactions or lets fraud slip through.
- **The NeuroGate Victory**: Evaluates semantic risk. When the margin between normal transfer and scam suspicion is borderline (`isAmbiguous`), it intercepts execution to trigger Step-Up 2FA (SMS OTP challenge), providing a dynamic middle-ground impossible in standard boolean control flow.

---

### 3. Executing Demonstrations

All demonstrations, domain suites, training compilers, and benchmarks are unified into the root [main.go](main.go) and specialized CLI tools:

```bash
# 1. Run Interactive Showcase (Shows Logo -> Scenario Menu -> On-The-Fly Train -> Live Branching)
./bin/neurogate-demo.exe
# Or directly via Go:
go run main.go

# 2. Direct Execution Mode (Skips Menu)
./bin/neurogate-demo.exe -mode domain -domain cs       # E-Commerce CS Gateway (XOR & Multi-Intent)
./bin/neurogate-demo.exe -mode domain -domain llm      # Semantic LLM Gateway & Cloud Bypass
./bin/neurogate-demo.exe -mode domain -domain sre      # High-Throughput SRE Log Triage (0 B/op)
./bin/neurogate-demo.exe -mode domain -domain iot      # Offline Edge IoT Control (Sub-180KB)
./bin/neurogate-demo.exe -mode domain -domain cicd     # Automated CI/CD Failure Triage
./bin/neurogate-demo.exe -mode domain -domain fintech  # FinTech Transaction Memo Audit (Step-up 2FA)
./bin/neurogate-demo.exe -mode router                  # High-Level 3-Tier Baseline Router
./bin/neurogate-demo.exe -mode bench                   # Zero-Allocation Latency Benchmark (~30 μs)
./bin/neurogate-demo.exe -mode all                     # Run All 6 Domains Sequentially
```

---

## Core Routing API Reference

| Method / Struct | Signature | Operational Role |
| :--- | :--- | :--- |
| **`NewRouter`** | `NewRouter(path string, threshold float64) (*Router, error)` | Loads v2 binary weights, initializes atomic model pointer, and builds 3-tier router. |
| **`SetPolicy`** | `.SetPolicy(policy DispatchPolicy) *Router` | Configures high/low thresholds, top-1/top-2 margin cutoff, OOD max entropy, and pipeline boundaries. |
| **`Bind`** | `.Bind(label string, handler RouteAction) *Router` | Associates a trained class with a Go handler: `func(ctx context.Context, payload any) error`. |
| **`BindPipeline`** | `.BindPipeline(p, s string, handler PipelineAction) *Router` | Registers composite handler triggered when primary and secondary intents are both eligible. |
| **`Ambiguous`** | `.Ambiguous(handler AmbiguousAction) *Router` | Intercepts borderline confidence or narrow margin queries to prompt user confirmation. |
| **`Fallback`** | `.Fallback(handler RouteAction) *Router` | Designates safety handler for low confidence, high unknown token ratio, or OOD entropy. |
| **`Dispatch`** | `.Dispatch(ctx context.Context, text string, payload any) error` | Evaluates 3-tier routing and executes bound branch in ~30 μs. |
| **`DispatchPipeline`**| `.DispatchPipeline(ctx context.Context, text string, payload any) error` | Executes multi-intent pipeline handlers if eligible, falling back to 3-tier routing. |
| **`Reload`** | `.Reload(path string) error` | Atomically swaps weights on live traffic without locks (`0 ns` stop-the-world). |
| **`EnableTelemetry`**| `.EnableTelemetry(capacity int) *Router` | Allocates thread-safe ring buffer capturing ambiguous, OOD, and pipeline requests. |
| **`DrainTelemetry`** | `.DrainTelemetry() []TelemetryEvent` | Extracts collected drift events in FIFO order for active learning retraining. |
| **`PredictSlots`** | `model.PredictSlots(text string, out *StaticInferenceResult) error` | Stack-allocated inference primitive achieving strictly **`0 B/op, 0 allocs/op`**. |

---

## Project Structure

```text
NeuroGate - demo/
├── bin/                   # Compiled single standalone binary
│   └── neurogate-demo.exe
├── data/                  # CSV datasets for 6 enterprise domains + quickstart
├── weights/               # Cached Little-Endian binary weights
├── main.go                # Unified training + inference lifecycle code
├── go.mod                 # Go module importing github.com/gluedays-cyber/neurogate
└── README.md              # Project technical manual & execution guide
```

---

## License

This project is licensed under the MIT License.
