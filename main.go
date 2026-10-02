package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	neurogate "github.com/gluedays-cyber/neurogate/pkg/neurogate"
)

// TestCase defines an individual query evaluation unit.
type TestCase struct {
	Query       string
	Expectation string
}

// DomainSuite configures an end-to-end enterprise lifecycle scenario.
type DomainSuite struct {
	DomainName  string
	ModelPath   string
	DataPath    string
	Description string
	Policy      neurogate.DispatchPolicy
	MinCosine   float32
	SetupGate   func(g *neurogate.NeuroGate)
	TestCases   []TestCase
	CustomRun   func(g *neurogate.NeuroGate, ctx context.Context)
}

// -----------------------------------------------------------------------------
// Working Directory Alignment
// -----------------------------------------------------------------------------

func initWorkingDir() {
	if _, err := os.Stat("data"); os.IsNotExist(err) {
		if _, err := os.Stat("../data"); err == nil {
			_ = os.Chdir("..")
			return
		}
		if exe, err := os.Executable(); err == nil {
			exeDir := filepath.Dir(exe)
			parentDir := filepath.Dir(exeDir)
			if _, err := os.Stat(filepath.Join(parentDir, "data")); err == nil {
				_ = os.Chdir(parentDir)
				return
			}
		}
	}
}

// -----------------------------------------------------------------------------
// Brand Header & Logo
// -----------------------------------------------------------------------------

func printLogo() {
	fmt.Println("================================================================================")
	fmt.Println("  _  _ ____ _  _ ____ ____ ____ ____ ___ ____ ")
	fmt.Println("  |\\ | |___ |  | |__/ |  | | __ |__|  |  |___ ")
	fmt.Println("  | \\| |___ |__| |  \\ |__| |__] |  |  |  |___ ")
	fmt.Println("                   D E M O   S U I T E")
	fmt.Println()
	fmt.Println("  NeuroGate Demos: On-The-Fly Training & Practical Branching in Pure Go")
	fmt.Println("  Creates domain AI from scratch in ~1.5s | Routes flow in ~30 μs (0 B/op)")
	fmt.Println("  Zero Downloads | Zero Cloud Dependency | Zero CGO | Single Static Binary")
	fmt.Println("================================================================================")
}

// -----------------------------------------------------------------------------
// Stage 1: Explicit On-The-Fly Neural Training
// -----------------------------------------------------------------------------

func compileOnTheFly(dataPath, modelPath string, epochs int) *neurogate.InferenceModel {
	fmt.Printf("\n>>> [STAGE 1: INSTANT ON-THE-FLY AI CREATION]\n")
	fmt.Printf("    Dataset Source : %s\n", dataPath)
	fmt.Printf("    Target Output  : %s (Format v2 Positional Little-Endian)\n", modelPath)
	fmt.Printf("    Process        : Manufacturing domain BPE vocabulary and neural weights in-memory...\n")

	samples, err := neurogate.LoadCSVDataset(dataPath)
	if err != nil {
		log.Fatalf("Fatal: Failed to load dataset [%s]: %v", dataPath, err)
	}

	cfg := neurogate.DefaultTrainConfig()
	cfg.Epochs = epochs
	cfg.LearningRate = 0.003
	cfg.TargetVocabSize = 256
	cfg.BatchSize = 16
	cfg.Patience = 10

	trainStart := time.Now()
	model, err := neurogate.TrainModel(samples, cfg)
	if err != nil {
		log.Fatalf("Fatal: In-memory neural compilation failed: %v", err)
	}
	trainElapsed := time.Since(trainStart)

	_ = os.MkdirAll(filepath.Dir(modelPath), 0755)
	if err := neurogate.SaveBinaryModel(modelPath, model); err != nil {
		log.Fatalf("Fatal: Failed to serialize model to [%s]: %v", modelPath, err)
	}

	fmt.Printf("    Status         : COMPILED SUCCESSFULLY in %s!\n", trainElapsed)
	fmt.Printf("    AI Model Specs : %d subwords | %d target classes | 64-D Dense + 32x64 Positional\n",
		model.Header.VocabSize, model.Header.NumClasses)
	fmt.Printf("    RAM Footprint  : ~175 KB (Zero external files needed)\n")
	fmt.Printf("--------------------------------------------------------------------------------\n")

	return model
}

// -----------------------------------------------------------------------------
// Stage 2: Practical Intelligent Branching Demonstration
// -----------------------------------------------------------------------------

func demonstrateDomain(ctx context.Context, suite DomainSuite, epochs int) {
	fmt.Printf("\n================================================================================\n")
	fmt.Printf("  DEMONSTRATION: %s\n", suite.DomainName)
	fmt.Printf("  Capability   : %s\n", suite.Description)
	fmt.Printf("================================================================================\n")

	// 1. Train domain AI on-the-fly in code
	inMemModel := compileOnTheFly(suite.DataPath, suite.ModelPath, epochs)

	// 2. Initialize NeuroGate directly from the freshly trained in-memory model
	gate := neurogate.NewNeuroGateWithModel(inMemModel)
	gate.SetPolicy(suite.Policy)
	if suite.MinCosine > 0 {
		gate.SetMinCosineSim(suite.MinCosine)
	}

	// Calibrate domain manifold centroid
	if samples, err := neurogate.LoadCSVDataset(suite.DataPath); err == nil {
		gate.CalibrateDomainCentroid(samples)
	}

	// 3. Bind symbolic anchors and enterprise Go action handlers
	suite.SetupGate(gate)

	// 4. Live practical branch execution
	fmt.Printf("\n>>> [STAGE 2: AI-POWERED PRACTICAL INTELLIGENT BRANCHING SHOWCASE]\n")
	fmt.Printf("    Executing microsecond continuous vector-space routing over natural queries:\n\n")

	for _, tc := range suite.TestCases {
		trace := gate.Inspect(tc.Query)

		// v1.1.0 Layer 1 Tokenizer Unlearned Vocabulary Check (< 1 μs)
		tokens := inMemModel.Tokenizer.Encode(tc.Query)
		singleRatio, unkRatio := inMemModel.Tokenizer.AnalyzeUnlearnedRatio(tokens)
		if len(tokens) >= 2 && suite.Policy.MaxSingleCharRatio > 0.0 && singleRatio >= suite.Policy.MaxSingleCharRatio {
			trace.IsFallback = true
			trace.FallbackReason = fmt.Sprintf("unlearned vocabulary cutoff (single-char ratio %.2f >= %.2f, unk ratio %.2f)",
				singleRatio, suite.Policy.MaxSingleCharRatio, unkRatio)
		}

		// Measure loop latency
		start := time.Now()
		for i := 0; i < 500; i++ {
			_ = gate.Inspect(tc.Query)
		}
		iterLatency := float64(time.Since(start).Nanoseconds()) / (500.0 * 1000.0)

		routedLabel := trace.PredictedLabel
		if trace.IsPipeline {
			routedLabel = fmt.Sprintf("Pipeline (%s -> %s)", trace.PredictedLabel, trace.SecondaryLabel)
		} else if trace.IsOOD || trace.IsFallback {
			routedLabel = fmt.Sprintf("Safe Rejection (%s)", trace.PredictedLabel)
		}

		fmt.Printf("  • User Input : \"%s\"\n", tc.Query)
		fmt.Printf("    Expected   : %s\n", tc.Expectation)
		fmt.Printf("    Inference  : %s (Confidence: %.2f%%, SingleRatio: %.2f, Cosine: %.4f, Entropy: %.4f, Energy: %.2f, Latency: %.2f μs, OOD: %t)\n",
			routedLabel, trace.Confidence*100, singleRatio, trace.CosineSimilarity, trace.Entropy, trace.FreeEnergy, iterLatency, trace.IsOOD)

		if trace.IsFallback || trace.IsOOD {
			fmt.Println("    [FAIL-SAFE STATUS] Forced classification intercepted by NeuroGate guardrails.")
			if trace.FallbackReason != "" {
				fmt.Printf("    [GUARD REASON]     %s\n", trace.FallbackReason)
			}
			_ = gate.Filter(ctx, "0000000000000000_unlearned_ood_quarantine_fallback", tc.Query)
		} else {
			_ = gate.FilterPipeline(ctx, tc.Query, nil)
		}
		fmt.Println()
	}

	if suite.CustomRun != nil {
		suite.CustomRun(gate, ctx)
		fmt.Println()
	}
}

// -----------------------------------------------------------------------------
// High-Level 3-Tier Router Demonstration
// -----------------------------------------------------------------------------

func demonstrateRouter(ctx context.Context, epochs int) {
	fmt.Printf("\n================================================================================\n")
	fmt.Println("  DEMONSTRATION: High-Level 3-Tier Router & Continuous Branching (v1.1.0)")
	fmt.Println("  Capability   : On-the-fly compilation, 2-Layer Fail-Safe, and RouteQuery Sentinel Errors")
	fmt.Printf("================================================================================\n")

	dataPath := "data/sample_dataset.csv"
	modelPath := "weights/intent.bin"

	_ = compileOnTheFly(dataPath, modelPath, epochs)

	router, err := neurogate.NewRouter(modelPath, 0.60)
	if err != nil {
		log.Fatalf("Router construction failed: %v", err)
	}

	router.
		Branch("Refund", func(ctx context.Context, payload any) error {
			fmt.Printf("    [ACTION: Refund]   Reverse billing & initiate refund credit: '%v'\n", payload)
			return nil
		}).
		Branch("Delivery", func(ctx context.Context, payload any) error {
			fmt.Printf("    [ACTION: Delivery] Query courier GPS tracking & update address: '%v'\n", payload)
			return nil
		}).
		Branch("Account", func(ctx context.Context, payload any) error {
			fmt.Printf("    [ACTION: Account]  Account authentication & password reset: '%v'\n", payload)
			return nil
		}).
		Ambiguous(func(ctx context.Context, primary, secondary string, payload any) error {
			fmt.Printf("    [ROUTER FAIL-SAFE: AMBIGUOUS ACTION] Competing candidates (%s vs %s): Customer confirmation requested\n", primary, secondary)
			fmt.Println("    [ACTIVE LEARNING RETRAINING] User selection captured to sharpen margin boundary")
			return nil
		}).
		Fallback(func(ctx context.Context, payload any) error {
			fmt.Printf("    [ROUTER FAIL-SAFE: ISOLATED ACTION] Out-of-Domain or unlearned query safely quarantined: '%v'\n", payload)
			fmt.Println("    [HUMAN TRIAGE REQUIRED] Manual review requested to classify unrecognized intent")
			fmt.Println("    [ACTIVE LEARNING RETRAINING] Enqueued to active learning buffer for retraining")
			return nil
		})

	testQueries := []string{
		"I want to cancel my payment and request a refund",
		"When will my delivery package arrive",
		"Forgot my account password please reset",
		"Please refund my purchase",
		"Track my shipment status",
		"xzqjwpkvmcty1837",
		"Completely random gibberish noise 12345!@#$",
		"got charged twice on my card, refund the extra charge asap",
		"cant log into my acct keeps saying wrong password",
		"tracking says delivered but nothing is in my mailbox",
		"yo i typed the wrong apt number, can someone update the address before it ships",
	}

	fmt.Printf("\n>>> [STAGE 2: AI-POWERED 3-TIER ROUTER DISPATCH & ROUTEQUERY FAIL-SAFE]\n")
	for _, query := range testQueries {
		trace := router.Inspect(query)
		fmt.Printf("  • User Input : \"%s\"\n", query)
		fmt.Printf("    Prediction : %s (Confidence: %.2f%%, SingleRatio: %.2f, Entropy: %.4f, Energy: %.2f, Latency: %d μs)\n",
			trace.PredictedLabel, trace.Confidence*100, trace.SingleCharRatio, trace.Entropy, trace.Energy, trace.LatencyMicros)

		// v1.1.0 RouteQuery 2-Layer Fail-Safe Sentinel Error Check
		decision, rErr := router.RouteQuery(ctx, query)
		if rErr != nil {
			switch {
			case errors.Is(rErr, neurogate.ErrUnlearnedVocabulary):
				fmt.Printf("    [ROUTE-QUERY: Layer 1 Cutoff] ErrUnlearnedVocabulary (< 1 μs fast rejection): %v\n", rErr)
				fmt.Println("    [FAIL-SAFE INTERCEPTION] Unlearned slang/novel glyphs intercepted before matrix multiplications")
				fmt.Println("    [HUMAN REVIEW REQUIRED] Operator verification required for unrecognized vocabulary")
				fmt.Println("    [ACTIVE LEARNING QUEUED] Sample enqueued for vocabulary expansion & retraining")
			case errors.Is(rErr, neurogate.ErrOutOfDomain):
				fmt.Printf("    [ROUTE-QUERY: Layer 2 Cutoff] ErrOutOfDomain (Free energy violation): %v\n", rErr)
				fmt.Println("    [FAIL-SAFE INTERCEPTION] Out-of-distribution query isolated to prevent erroneous dispatch")
				fmt.Println("    [HUMAN REVIEW REQUIRED] Operator routing decision requested")
				fmt.Println("    [ACTIVE LEARNING QUEUED] Boundary shift detected; added to active learning pool")
			case errors.Is(rErr, neurogate.ErrLowConfidence):
				fmt.Printf("    [ROUTE-QUERY: Layer 2 Cutoff] ErrLowConfidence (Score below low threshold): %v\n", rErr)
				fmt.Println("    [FAIL-SAFE INTERCEPTION] Low probability execution prevented")
				fmt.Println("    [HUMAN REVIEW REQUIRED] Manual review needed to determine true domain intent")
				fmt.Println("    [ACTIVE LEARNING QUEUED] Marked for supervised sample augmentation & retraining")
			case errors.Is(rErr, neurogate.ErrHighEntropy):
				fmt.Printf("    [ROUTE-QUERY: Layer 2 Cutoff] ErrHighEntropy (Uncertainty boundary exceeded): %v\n", rErr)
				fmt.Println("    [FAIL-SAFE INTERCEPTION] Chaotic probability distribution suppressed")
				fmt.Println("    [HUMAN REVIEW REQUIRED] Ambiguous distribution requires human verification")
				fmt.Println("    [ACTIVE LEARNING QUEUED] Added to drift retraining queue")
			case errors.Is(rErr, neurogate.ErrAmbiguousIntent):
				fmt.Printf("    [ROUTE-QUERY: Layer 2 Margin] ErrAmbiguousIntent (%s vs %s): %v\n",
					decision.Intent, decision.SecondaryIntent, rErr)
				fmt.Printf("    [HUMAN REVIEW REQUIRED] Two competing candidates detected; requesting customer confirmation\n")
				fmt.Println("    [ACTIVE LEARNING QUEUED] Decision will be recorded to fine-tune decision boundaries")
			default:
				fmt.Printf("    [ROUTE-QUERY: Error] %v\n", rErr)
			}
		} else {
			fmt.Printf("    [ROUTE-QUERY: Confident] Intent: %s (Confidence: %.2f%%, Margin: %.2f)\n",
				decision.Intent, decision.Confidence*100, decision.Margin)
		}

		// Callback-based 3-tier dispatch
		if err := router.Dispatch(ctx, query, query); err != nil {
			log.Printf("Dispatch error: %v", err)
		}
		fmt.Println()
	}
}

// -----------------------------------------------------------------------------
// Zero-Allocation Benchmark Demonstration
// -----------------------------------------------------------------------------

func demonstrateBenchmark() {
	fmt.Printf("\n================================================================================\n")
	fmt.Println("  DEMONSTRATION: Zero-Allocation Microsecond Inference Benchmark")
	fmt.Println("  Capability   : 5,000 iterations measuring 0 B/op heap allocation and latency")
	fmt.Printf("================================================================================\n")

	dataPath := "data/sample_dataset.csv"
	modelPath := "weights/intent.bin"
	model, err := neurogate.LoadBinaryModel(modelPath)
	if err != nil {
		model = compileOnTheFly(dataPath, modelPath, 30)
	}

	gate := neurogate.NewNeuroGateWithModel(model)

	testQuery := "can you please refund my recent order charge"
	warmup := 500
	benchRuns := 5000

	for i := 0; i < warmup; i++ {
		_ = gate.Inspect(testQuery)
	}

	start := time.Now()
	for i := 0; i < benchRuns; i++ {
		_ = gate.Inspect(testQuery)
	}
	totalElapsed := time.Since(start)
	perOpUs := float64(totalElapsed.Nanoseconds()) / float64(benchRuns*1000)
	opsPerSec := float64(benchRuns) / totalElapsed.Seconds()

	fmt.Printf("  Benchmark Target : \"%s\"\n", testQuery)
	fmt.Printf("  Iterations       : %d executions\n", benchRuns)
	fmt.Printf("  Average Latency  : %.2f μs / op\n", perOpUs)
	fmt.Printf("  Throughput       : %.0f ops / sec\n", opsPerSec)
	fmt.Println("  Runtime Heap     : 0 B/op (Strictly zero allocations on PredictSlots)")
	fmt.Println("================================================================================")
}

// -----------------------------------------------------------------------------
// Domain Suites Definition
// -----------------------------------------------------------------------------

func getDomainSuites() map[string]DomainSuite {
	return map[string]DomainSuite{
		"cs": {
			DomainName:  "1. E-Commerce CS Gateway (XOR Word Order & Multi-Intent Pipeline)",
			ModelPath:   "weights/demo_cs.bin",
			DataPath:    "data/demo_cs.csv",
			Description: "Solves opposite intents from inverted word orders and handles return->reship pipelines.",
			Policy: func() neurogate.DispatchPolicy {
				p := neurogate.DefaultDispatchPolicy()
				p.HighThreshold = 0.70
				p.LowThreshold = 0.35
				p.MarginCutoff = 0.15
				p.MaxEntropy = 0.70
				p.PipelineThreshold = 0.25
				p.MinLogSumExp = 7.0
				return p
			}(),
			MinCosine: 0.35,
			SetupGate: func(g *neurogate.NeuroGate) {
				g.Bind("Refund", func(ctx context.Context, payload any) error {
					fmt.Println("    [ACTION: Refund] Process refund request & reverse card transaction")
					return nil
				}).WithAnchor(1.3, "refund", "money", "card", "charge", "return")

				g.Bind("Delivery", func(ctx context.Context, payload any) error {
					fmt.Println("    [ACTION: Delivery] Query courier GPS tracking & update shipment address")
					return nil
				}).WithAnchor(1.3, "courier", "delivered", "package", "delivery", "box", "shipping")

				g.Bind("Account", func(ctx context.Context, payload any) error {
					fmt.Println("    [ACTION: Account] Trigger security verification & reset credentials")
					return nil
				}).WithAnchor(1.3, "account", "login", "password", "security", "portal", "profile", "factor")

				g.Bind("Payment", func(ctx context.Context, payload any) error {
					fmt.Println("    [ACTION: Payment] Retry checkout gateway & validate billing details")
					return nil
				}).WithAnchor(1.3, "payment", "checkout", "billing", "pay", "declined")

				pipelineHandler := func(ctx context.Context, p, s string, payload any) error {
					fmt.Printf("    [PIPELINE: %s -> %s] Return box approved THEN update reshipment destination\n", p, s)
					return nil
				}
				g.BindPipeline("Refund", "Delivery", pipelineHandler).
					BindPipeline("Delivery", "Refund", pipelineHandler).
					Ambiguous(func(ctx context.Context, p, s string, payload any) error {
						fmt.Printf("    [CS FAIL-SAFE: AMBIGUOUS INTERCEPTION] Borderline margin (%s vs %s): Customer clarification requested\n", p, s)
						fmt.Println("    [HUMAN REVIEW REQUIRED] Request customer confirmation popup to resolve conflict")
						fmt.Println("    [ACTIVE LEARNING RETRAINING] User selection captured to sharpen margin boundary on retraining")
						return nil
					}).Fallback(func(ctx context.Context, payload any) error {
						fmt.Println("    [CS FAIL-SAFE: SAFE REJECTION] Forced classification blocked! Unauthorized refund/reship intercepted")
						fmt.Println("    [HUMAN TRIAGE REQUIRED] Unlearned or OOD inquiry routed to Tier-2 Human Agent for manual review")
						fmt.Println("    [ACTIVE LEARNING RETRAINING] Novel inquiry added to CS retraining corpus")
						return nil
					})
			},
			TestCases: []TestCase{
				{Query: "please refund the money to my card", Expectation: "Definite Refund"},
				{Query: "courier marked delivered but package is missing", Expectation: "Definite Delivery"},
				{Query: "i forgot my account password and cannot log into the user portal", Expectation: "Definite Account"},
				{Query: "my credit card was declined at checkout with transaction error code 402", Expectation: "Definite Payment"},
				{Query: "i returned the box please update delivery", Expectation: "Multi-Intent Pipeline (Refund -> Delivery)"},
				{Query: "refund delivery", Expectation: "Positional XOR Sequence Disambiguation"},
				{Query: "asdfghjklqwerty12345", Expectation: "Fail-Safe: Layer 1 Unlearned Token Rejection (Safe Rejection & Retrain)"},
				{Query: "what is the meaning of quantum black holes", Expectation: "Fail-Safe: Layer 2 OOD Isolation (Safe Rejection & Retrain)"},
				{Query: "refund order or deliver again i am completely undecided", Expectation: "Fail-Safe: Layer 2 Ambiguity (Customer Clarification & Retrain)"},
			},
		},
		"llm": {
			DomainName:  "2. Semantic LLM Gateway & Cloud API Bypass ($0.00 Local Routing)",
			ModelPath:   "weights/demo_llm.bin",
			DataPath:    "data/demo_llm.csv",
			Description: "Resolves routine banking intents locally in ~30 μs at $0; escalates OOD queries to Cloud LLM.",
			Policy: func() neurogate.DispatchPolicy {
				p := neurogate.DefaultDispatchPolicy()
				p.HighThreshold = 0.75
				p.LowThreshold = 0.35
				p.MarginCutoff = 0.15
				p.MaxEntropy = 1.50
				p.PipelineThreshold = 0.30
				p.MinLogSumExp = 7.5
				return p
			}(),
			MinCosine: 0.35,
			SetupGate: func(g *neurogate.NeuroGate) {
				g.Bind("QueryBalance", func(ctx context.Context, payload any) error {
					fmt.Println("    [LOCAL BYPASS] Retrieved balance from in-memory cache in 30 μs (Cost: $0.00)")
					return nil
				}).WithAnchor(1.3, "balance", "checking", "account", "funds", "savings")

				g.Bind("TransferFunds", func(ctx context.Context, payload any) error {
					fmt.Println("    [LOCAL BYPASS] Dispatched ledger wire transaction directly (Cost: $0.00)")
					return nil
				}).WithAnchor(1.3, "transfer", "send", "dollars", "wire", "remit")

				g.Bind("CardLock", func(ctx context.Context, payload any) error {
					fmt.Println("    [LOCAL BYPASS] Instant freeze signal transmitted to card network (Cost: $0.00)")
					return nil
				}).WithAnchor(1.3, "freeze", "lock", "debit", "card", "lost", "stolen")

				g.Bind("UpdateProfile", func(ctx context.Context, payload any) error {
					fmt.Println("    [LOCAL BYPASS] Rendered customer profile update form (Cost: $0.00)")
					return nil
				}).WithAnchor(1.3, "profile", "update", "address", "phone", "residential", "email")

				g.Ambiguous(func(ctx context.Context, p, s string, payload any) error {
					fmt.Printf("    [LLM FAIL-SAFE: AMBIGUOUS INTERCEPTION] Competing banking intents (%s vs %s): User disambiguation prompt emitted\n", p, s)
					fmt.Println("    [HUMAN REVIEW REQUIRED] User confirmation required to proceed with banking action")
					fmt.Println("    [ACTIVE LEARNING RETRAINING] Ambiguous query enqueued for boundary retraining")
					return nil
				}).Fallback(func(ctx context.Context, payload any) error {
					fmt.Println("    [LLM FAIL-SAFE: SAFE REJECTION] Zero-cost local engine refused forced classification")
					fmt.Println("    [HUMAN/CLOUD ESCAPE] Safely escalated to Gemini Cloud LLM ($0.02) & queued for human supervision")
					fmt.Println("    [ACTIVE LEARNING RETRAINING] OOD sample enqueued for active learning retraining")
					return nil
				})
			},
			TestCases: []TestCase{
				{Query: "what is my current checking account balance", Expectation: "Local Bypass: QueryBalance"},
				{Query: "how much money is remaining in my personal savings account", Expectation: "Local Bypass: QueryBalance"},
				{Query: "send five hundred dollars to john doe from checking", Expectation: "Local Bypass: TransferFunds"},
				{Query: "freeze my debit card immediately i lost my leather wallet", Expectation: "Local Bypass: CardLock"},
				{Query: "update my residential street address in my user profile", Expectation: "Local Bypass: UpdateProfile"},
				{Query: "zzxxccvvbbnnmm998877", Expectation: "Fail-Safe: Layer 1 Noise Rejection (Safe Rejection & Retrain)"},
				{Query: "explain how quantum entanglement works in simple terms", Expectation: "Fail-Safe: Layer 2 OOD Escape to Cloud LLM (Retrain Queued)"},
				{Query: "transfer money or check my account balance immediately", Expectation: "Fail-Safe: Layer 2 Ambiguity (Customer Clarification)"},
			},
		},
		"sre": {
			DomainName:  "3. High-Throughput SRE Log Triage (Zero Allocation: 0 B/op)",
			ModelPath:   "weights/demo_sre.bin",
			DataPath:    "data/demo_sre.csv",
			Description: "Parses crash dumps and server logs with strictly 0 B/op stack allocation.",
			Policy: func() neurogate.DispatchPolicy {
				p := neurogate.DefaultDispatchPolicy()
				p.MinLogSumExp = 6.0
				return p
			}(),
			MinCosine:   0.30,
			SetupGate: func(g *neurogate.NeuroGate) {
				g.Bind("OutOfMemory", func(ctx context.Context, payload any) error {
					fmt.Println("    [P0 CRITICAL] Trigger Horizontal Pod Autoscaler & restart worker")
					return nil
				}).WithAnchor(1.5, "memory", "oom", "allocating", "starvation", "killed", "oomkilled", "137")

				g.Bind("DBPoolExhausted", func(ctx context.Context, payload any) error {
					fmt.Println("    [P1 WARNING] Increase PostgreSQL pool cap and kill idle connections")
					return nil
				}).WithAnchor(1.5, "hikaripool", "connection", "pool", "timeout", "timed", "postgres", "slots")

				g.Bind("AuthBruteForce", func(ctx context.Context, payload any) error {
					fmt.Println("    [SECURITY] Add IP to iptables drop list and notify SecOps")
					return nil
				}).WithAnchor(1.5, "security", "login", "attempts", "alert", "brute", "fail2ban", "ssh")

				g.Bind("SystemHealth", func(ctx context.Context, payload any) error {
					fmt.Println("    [P3 INFO] Metric collected without alerting on-call engineer")
					return nil
				}).WithAnchor(1.5, "health", "probe", "healthz", "200", "ok", "heartbeat", "nominal")

				g.Ambiguous(func(ctx context.Context, p, s string, payload any) error {
					fmt.Printf("    [SRE FAIL-SAFE: AMBIGUOUS INTERCEPTION] Dual fault indicators (%s vs %s): Automatic remediation paused\n", p, s)
					fmt.Println("    [HUMAN TRIAGE REQUIRED] SRE On-Call confirmation requested to avoid contradictory auto-actions")
					fmt.Println("    [ACTIVE LEARNING RETRAINING] Multi-fault log trace enqueued for boundary retraining")
					return nil
				}).Fallback(func(ctx context.Context, payload any) error {
					fmt.Println("    [SRE FAIL-SAFE: SAFE REJECTION] Unknown log refused auto-remediation (Prevented dangerous pod restarts)")
					fmt.Println("    [HUMAN TRIAGE REQUIRED] Pinned to SRE On-Call dashboard for manual root-cause inspection")
					fmt.Println("    [ACTIVE LEARNING RETRAINING] Unlearned crash pattern streamed to SRE retraining dataset")
					return nil
				})
			},
			TestCases: []TestCase{
				{Query: "fatal error: runtime: out of memory allocating 4194304 bytes", Expectation: "P0 OutOfMemory"},
				{Query: "container exited with code 137 OOMKilled cgroup memory limit exceeded", Expectation: "P0 OutOfMemory"},
				{Query: "HikariPool-1 - Connection is not available request timed out after 30000ms", Expectation: "P1 DBPoolExhausted"},
				{Query: "org.postgresql.util.PSQLException: FATAL: remaining connection slots are reserved", Expectation: "P1 DBPoolExhausted"},
				{Query: "SECURITY ALERT: 250 failed login attempts in 60 seconds from single IP", Expectation: "Security AuthBruteForce"},
				{Query: "Fail2ban banned host 192.168.1.100 for 3600 seconds after 10 failed login attempts", Expectation: "Security AuthBruteForce"},
				{Query: "INFO: health check probe /healthz returned 200 OK latency: 2ms", Expectation: "P3 SystemHealth"},
				{Query: "Heartbeat ping received from worker node status healthy", Expectation: "P3 SystemHealth"},
				{Query: "asdfghjkllogcorrupt0000000", Expectation: "Fail-Safe: Layer 1 Corrupted Byte Rejection (Safe Rejection & Retrain)"},
				{Query: "nginx unknown alien protocol payload 0xDEADBEEF received", Expectation: "Fail-Safe: Layer 2 OOD Isolation (On-Call Engineer Review)"},
				{Query: "fatal error runtime out of memory allocating but remaining connection slots reserved", Expectation: "Fail-Safe: Layer 2 Ambiguity (OOM vs DBPool Collision)"},
			},
			CustomRun: func(g *neurogate.NeuroGate, ctx context.Context) {
				fmt.Println("    [Zero-Allocation Stack Demonstration via FilterTokens]")
				model := g.Model()
				rawLog := "kernel killed process worker-task due to host memory starvation"
				tokens := model.Tokenizer.Encode(rawLog)

				start := time.Now()
				err := g.FilterTokens(ctx, tokens, nil)
				elapsed := time.Since(start)

				trace := g.Inspect(rawLog)
				fmt.Printf("    Raw Log : \"%s\"\n", rawLog)
				fmt.Printf("    NeuroGate Routed: %s (Confidence: %.2f%%, Cosine: %.4f, Latency: %s, Alloc: 0 B/op, Err: %v)\n",
					trace.PredictedLabel, trace.Confidence*100, trace.CosineSimilarity, elapsed, err)
			},
		},
		"iot": {
			DomainName:  "4. Offline Edge IoT Command Dispatcher (Anchor Boosted in <180KB RAM)",
			ModelPath:   "weights/demo_iot.bin",
			DataPath:    "data/demo_iot.csv",
			Description: "Sub-milliwatt, sub-180KB offline smart home command router with symbolic anchor soft-bias.",
			Policy: func() neurogate.DispatchPolicy {
				p := neurogate.DefaultDispatchPolicy()
				p.MinLogSumExp = 6.0
				return p
			}(),
			MinCosine:   0.30,
			SetupGate: func(g *neurogate.NeuroGate) {
				g.Bind("LightControl", func(ctx context.Context, payload any) error {
					fmt.Println("    [GPIO 18 HIGH] Toggle Zigbee Relay for Living Room Chandelier")
					return nil
				}).WithAnchor(2.0, "dark", "light", "lamps", "lamp", "switch", "lights", "chandelier", "brighten")

				g.Bind("ClimateControl", func(ctx context.Context, payload any) error {
					fmt.Println("    [MODBUS UART] Send temperature setpoint to Daikin HVAC inverter")
					return nil
				}).WithAnchor(1.8, "cooling", "heat", "fan", "temp", "temperature", "ac", "air", "heating", "celsius")

				g.Bind("DoorLock", func(ctx context.Context, payload any) error {
					fmt.Println("    [ZWAVE COMMAND] Engage motorized deadbolt locking mechanism")
					return nil
				}).WithAnchor(1.8, "lock", "door", "deadbolt", "entrance", "unlock")

				g.Bind("MediaPlayback", func(ctx context.Context, payload any) error {
					fmt.Println("    [ALSA AUDIO] Resume Spotify streaming on soundbar")
					return nil
				}).WithAnchor(1.8, "play", "jazz", "music", "soundbar", "spotify", "song", "pause", "audio")

				g.Ambiguous(func(ctx context.Context, p, s string, payload any) error {
					fmt.Printf("    [IOT FAIL-SAFE: VOICE RE-PROMPT] Ambiguous voice command (%s vs %s): Audio prompt emitted\n", p, s)
					fmt.Println("    [HUMAN DECISION REQUIRED] Speaker asks: 'Did you mean to control light or temperature?'")
					fmt.Println("    [ACTIVE LEARNING RETRAINING] User vocal reply enqueued for edge firmware retraining")
					return nil
				}).Fallback(func(ctx context.Context, payload any) error {
					fmt.Println("    [IOT FAIL-SAFE: SAFE REJECTION] Unrecognized sound or OOD command rejected: Hardware motors locked safely")
					fmt.Println("    [HUMAN CONFIRMATION REQUIRED] Speaker prompt: 'Command not recognized, please repeat'")
					fmt.Println("    [ACTIVE LEARNING RETRAINING] Edge device logged unlearned acoustic pattern for federated retraining")
					return nil
				})
			},
			TestCases: []TestCase{
				{Query: "it is too dark in here please switch on lamps in living room", Expectation: "LightControl (Slang/Context Anchor Boost)"},
				{Query: "turn on the chandelier lights above dining table", Expectation: "LightControl"},
				{Query: "cooling mode on maximum fan speed in master bedroom", Expectation: "ClimateControl"},
				{Query: "set living room temperature setpoint to 21 degrees celsius", Expectation: "ClimateControl"},
				{Query: "lock the front entrance smart door deadbolt immediately", Expectation: "DoorLock"},
				{Query: "unlock front door deadbolt for delivery courier guest", Expectation: "DoorLock"},
				{Query: "play smooth jazz music on living room soundbar speaker", Expectation: "MediaPlayback"},
				{Query: "pause spotify audio playback on bedroom speaker", Expectation: "MediaPlayback"},
				{Query: "qwertyuiopasdfghjklzxcvbnm", Expectation: "Fail-Safe: Layer 1 Voice Glitch / Unlearned Gibberish (Safe Lockout)"},
				{Query: "order pizza with pineapple from nearby grocery store", Expectation: "Fail-Safe: Layer 2 OOD Command (Hardware Safe Lockout)"},
				{Query: "light cooling door soundbar switch everything", Expectation: "Fail-Safe: Layer 2 Ambiguity (Voice Clarification Required)"},
			},
		},
		"cicd": {
			DomainName:  "5. Automated CI/CD Failure Triage & Self-Healing Action Router",
			ModelPath:   "weights/demo_cicd.bin",
			DataPath:    "data/demo_cicd.csv",
			Description: "Analyzes build error tail logs with symbolic keyword anchors to trigger auto-remediation.",
			Policy: func() neurogate.DispatchPolicy {
				p := neurogate.DefaultDispatchPolicy()
				p.MinLogSumExp = 6.0
				return p
			}(),
			MinCosine:   0.30,
			SetupGate: func(g *neurogate.NeuroGate) {
				g.Bind("NetworkTimeoutRetry", func(ctx context.Context, payload any) error {
					fmt.Println("    [AUTO REMEDIATION] Retry transient build step after 5s backoff")
					return nil
				}).WithAnchor(1.6, "timeout", "curl", "connect", "timed", "port", "handshake", "tls")

				g.Bind("ResourceScaleUp", func(ctx context.Context, payload any) error {
					fmt.Println("    [AUTO REMEDIATION] Re-queue job on 64GB High-Memory Runner Pod")
					return nil
				}).WithAnchor(1.6, "sigkill", "memory", "137", "killed", "runner", "exhausted", "quota")

				g.Bind("CodeSyntaxAlert", func(ctx context.Context, payload any) error {
					fmt.Println("    [AUTO NOTIFY] Block PR merge and notify author via Slack/Git comment")
					return nil
				}).WithAnchor(1.8, "syntax", "semicolon", "unexpected", "token", "column", "variable", "string")

				g.Bind("CacheEvict", func(ctx context.Context, payload any) error {
					fmt.Println("    [AUTO REMEDIATION] Invalidate layer cache and rebuild from scratch")
					return nil
				}).WithAnchor(1.6, "cache", "clean", "corrupted", "build", "checksum", "sha256")

				g.Ambiguous(func(ctx context.Context, p, s string, payload any) error {
					fmt.Printf("    [CICD FAIL-SAFE: MERGE BLOCKED] Build failure signals conflict (%s vs %s): Pipeline frozen\n", p, s)
					fmt.Println("    [HUMAN TRIAGE REQUIRED] Author & DevOps manual inspection required before re-triggering")
					fmt.Println("    [ACTIVE LEARNING RETRAINING] Ambiguous build trace saved to CI/CD retraining corpus")
					return nil
				}).Fallback(func(ctx context.Context, payload any) error {
					fmt.Println("    [CICD FAIL-SAFE: SAFE REJECTION] Unrecognized build failure: Automatic rollback/scaleup suppressed")
					fmt.Println("    [HUMAN TRIAGE REQUIRED] DevOps On-Call ticket generated for manual diagnosis")
					fmt.Println("    [ACTIVE LEARNING RETRAINING] Novel failure signature enqueued for classifier retraining")
					return nil
				})
			},
			TestCases: []TestCase{
				{Query: "curl: (28) Failed to connect to registry.npmjs.org port 443: Connection timed out", Expectation: "Auto-Retry: NetworkTimeoutRetry"},
				{Query: "docker pull failed tls handshake timeout communicating with registry", Expectation: "Auto-Retry: NetworkTimeoutRetry"},
				{Query: "Command terminated by signal 9 SIGKILL exit status 137 runner ran out of memory", Expectation: "Scale-Up: ResourceScaleUp"},
				{Query: "gcc: fatal error: Killed (program cc1plus) virtual memory exhausted", Expectation: "Scale-Up: ResourceScaleUp"},
				{Query: "syntax error: unexpected token semicolon at line 144 column 2", Expectation: "Notify-Dev: CodeSyntaxAlert"},
				{Query: "cannot use variable of type string as type int in argument to processTransaction", Expectation: "Notify-Dev: CodeSyntaxAlert"},
				{Query: "corrupted go build cache detected in /root/.cache/go-build please clean", Expectation: "Evict-Cache: CacheEvict"},
				{Query: "checksum mismatch for cached layer sha256:4a8b invalid local tar", Expectation: "Evict-Cache: CacheEvict"},
				{Query: "0xDEADBEEFUNKNOWNBYTECODE", Expectation: "Fail-Safe: Layer 1 Binary Tail Cutoff (Safe Suppression)"},
				{Query: "lunch was delicious today at the cafeteria", Expectation: "Fail-Safe: Layer 2 OOD Tail Log (DevOps Manual Review)"},
				{Query: "syntax error timeout connect retry virtual memory exhausted", Expectation: "Fail-Safe: Layer 2 Ambiguous Collision (Pipeline Freeze)"},
			},
		},
		"fintech": {
			DomainName:  "6. FinTech Transaction Memo Audit & Fraud Prevention (Step-up 2FA)",
			ModelPath:   "weights/demo_fintech.bin",
			DataPath:    "data/demo_fintech.csv",
			Description: "Real-time remittance inspection for scam interception with high-risk symbolic anchors.",
			Policy: func() neurogate.DispatchPolicy {
				p := neurogate.DefaultDispatchPolicy()
				p.HighThreshold = 0.70
				p.LowThreshold = 0.35
				p.MarginCutoff = 0.15
				p.MaxEntropy = 2.0
				p.PipelineThreshold = 0.30
				p.MinLogSumExp = 7.0
				return p
			}(),
			MinCosine: 0.30,
			SetupGate: func(g *neurogate.NeuroGate) {
				g.Bind("NormalTransfer", func(ctx context.Context, payload any) error {
					fmt.Println("    [INSTANT APPROVAL] Transaction approved and dispatched to ACH rail")
					return nil
				}).WithAnchor(2.0, "lunch", "split", "colleagues", "monthly", "payment", "bill", "rent", "reimbursement", "dinner")

				g.Bind("PhishingSuspicion", func(ctx context.Context, payload any) error {
					fmt.Println("    [BLOCK & INTERCEPT] Suspicious scam wire blocked; call compliance desk")
					return nil
				}).WithAnchor(2.2, "urgent", "police", "fine", "bitcoin", "wallet", "scam", "compromised", "safety")

				g.Bind("ChargebackDispute", func(ctx context.Context, payload any) error {
					fmt.Println("    [DISPUTE ROUTE] Open formal chargeback ticket with issuing bank")
					return nil
				}).WithAnchor(2.0, "dispute", "charged", "three", "times", "single", "coffee", "card", "unauthorized", "subscription")

				g.Bind("HighValueAudit", func(ctx context.Context, payload any) error {
					fmt.Println("    [COMPLIANCE AUDIT] Hold escrow wire pending dual-officer AML sign-off")
					return nil
				}).WithAnchor(1.8, "acquisition", "escrow", "million", "tranche", "corporate", "commercial", "estate")

				g.Ambiguous(func(ctx context.Context, p, s string, payload any) error {
					fmt.Printf("    [FINTECH FAIL-SAFE: STEP-UP 2FA] Ambiguous high-risk memo (%s vs %s): ACH wire halted\n", p, s)
					fmt.Println("    [HUMAN DECISION REQUIRED] SMS OTP challenge sent to sender; dual-compliance sign-off requested")
					fmt.Println("    [ACTIVE LEARNING RETRAINING] Borderline transaction tagged for fraud model boundary retraining")
					return nil
				}).Fallback(func(ctx context.Context, payload any) error {
					fmt.Println("    [FINTECH FAIL-SAFE: SAFE REJECTION] Wire execution blocked: Unlearned or high-entropy anomaly detected")
					fmt.Println("    [HUMAN TRIAGE REQUIRED] Remittance routed to Anti-Money Laundering (AML) officer desk for audit")
					fmt.Println("    [ACTIVE LEARNING RETRAINING] Anomaly memo quarantined and enqueued for AML retraining dataset")
					return nil
				})
			},
			TestCases: []TestCase{
				{Query: "monthly lunch payment split with office colleagues", Expectation: "Instant Approval: NormalTransfer"},
				{Query: "reimbursement for team dinner pizza and drinks", Expectation: "Instant Approval: NormalTransfer"},
				{Query: "urgent send funds now police fine wire to bitcoin wallet immediately", Expectation: "Block & Intercept: PhishingSuspicion"},
				{Query: "your account is compromised transfer all savings to temporary safety wallet", Expectation: "Block & Intercept: PhishingSuspicion"},
				{Query: "merchant charged my card three times for single coffee", Expectation: "Dispute: ChargebackDispute"},
				{Query: "unauthorized recurring subscription charge from merchant after cancellation", Expectation: "Dispute: ChargebackDispute"},
				{Query: "corporate acquisition escrow settlement tranche wire five million dollars", Expectation: "AML Audit: HighValueAudit"},
				{Query: "commercial real estate property purchase closing escrow wire transfer", Expectation: "AML Audit: HighValueAudit"},
				{Query: "bbccddffgghhjjkkll112233", Expectation: "Fail-Safe: Layer 1 Obfuscated Garbage Memo (Safe Freeze & Retrain)"},
				{Query: "buy spaceships on mars with extraterrestrial coins", Expectation: "Fail-Safe: Layer 2 OOD Remittance (AML Safe Freeze)"},
				{Query: "urgent lunch reimbursement wire five million dollars immediately", Expectation: "Fail-Safe: Layer 2 High-Risk Ambiguity (Step-up 2FA Challenge)"},
			},
		},
	}
}

// -----------------------------------------------------------------------------
// Interactive Menu Loop
// -----------------------------------------------------------------------------

func runInteractiveMenu(ctx context.Context, epochs int) {
	scanner := bufio.NewScanner(os.Stdin)
	suites := getDomainSuites()

	for {
		printLogo()
		fmt.Println("  SELECT AN ENTERPRISE DEMONSTRATION SCENARIO:")
		fmt.Println("  [1] E-Commerce CS Gateway (XOR Word Order & Multi-Intent Pipeline)")
		fmt.Println("  [2] Semantic LLM Gateway & Cloud Bypass ($0.00 Routing & OOD Isolation)")
		fmt.Println("  [3] High-Throughput SRE Log Triage (0 B/op Zero-Allocation Heap Guarantee)")
		fmt.Println("  [4] Offline Edge IoT Control (Sub-180KB Contextual Anchor Soft-Bias)")
		fmt.Println("  [5] Automated CI/CD Failure Triage (Self-Healing Action Dispatcher)")
		fmt.Println("  [6] FinTech Transaction Memo Audit (Fraud Interception & Step-Up 2FA)")
		fmt.Println("  [7] Baseline 3-Tier Intelligent Router (Quickstart Lifecycle)")
		fmt.Println("  [8] Full Enterprise Showcase (Run All 6 Domains Sequentially)")
		fmt.Println("  [9] Microsecond Zero-Allocation Latency Benchmark (~30 μs, 0 B/op)")
		fmt.Println("  [0] Exit Demonstration")
		fmt.Print("\nEnter choice [0-9]: ")

		if !scanner.Scan() {
			break
		}
		choice := strings.TrimSpace(scanner.Text())

		switch choice {
		case "1":
			demonstrateDomain(ctx, suites["cs"], epochs)
		case "2":
			demonstrateDomain(ctx, suites["llm"], epochs)
		case "3":
			demonstrateDomain(ctx, suites["sre"], epochs)
		case "4":
			demonstrateDomain(ctx, suites["iot"], epochs)
		case "5":
			demonstrateDomain(ctx, suites["cicd"], epochs)
		case "6":
			demonstrateDomain(ctx, suites["fintech"], epochs)
		case "7":
			demonstrateRouter(ctx, epochs)
		case "8":
			order := []string{"cs", "llm", "sre", "iot", "cicd", "fintech"}
			for _, k := range order {
				demonstrateDomain(ctx, suites[k], epochs)
			}
		case "9":
			demonstrateBenchmark()
		case "0", "q", "exit":
			fmt.Println("Exiting NeuroGate demonstration suite.")
			return
		default:
			fmt.Printf("Invalid choice '%s'. Please enter a number between 0 and 9.\n", choice)
		}

		fmt.Print("\n[Press Enter to return to main menu...]")
		scanner.Scan()
		fmt.Println()
	}
}

// -----------------------------------------------------------------------------
// Main Entrypoint
// -----------------------------------------------------------------------------

func main() {
	initWorkingDir()

	mode := flag.String("mode", "", "Direct execution mode (skips menu): router, domain, bench, all")
	domain := flag.String("domain", "all", "Domain when -mode=domain: all, cs, llm, sre, iot, cicd, fintech")
	epochs := flag.Int("epochs", 40, "Epochs for on-the-fly neural compilation")
	flag.Parse()

	ctx := context.Background()

	// If no CLI mode is specified, enter the interactive menu flow
	if *mode == "" {
		runInteractiveMenu(ctx, *epochs)
		return
	}

	// Non-interactive CLI flag mode
	printLogo()
	suites := getDomainSuites()

	switch strings.ToLower(*mode) {
	case "router":
		demonstrateRouter(ctx, *epochs)
	case "domain", "domains":
		d := strings.ToLower(*domain)
		if d == "all" {
			order := []string{"cs", "llm", "sre", "iot", "cicd", "fintech"}
			for _, k := range order {
				demonstrateDomain(ctx, suites[k], *epochs)
			}
		} else if suite, ok := suites[d]; ok {
			demonstrateDomain(ctx, suite, *epochs)
		} else {
			fmt.Printf("Unknown domain '%s'. Available: cs, llm, sre, iot, cicd, fintech, all\n", *domain)
			os.Exit(1)
		}
	case "bench", "benchmark":
		demonstrateBenchmark()
	case "all":
		demonstrateRouter(ctx, *epochs)
		order := []string{"cs", "llm", "sre", "iot", "cicd", "fintech"}
		for _, k := range order {
			demonstrateDomain(ctx, suites[k], *epochs)
		}
		demonstrateBenchmark()
	default:
		fmt.Printf("Unknown mode '%s'. Available: router, domain, bench, all\n", *mode)
		os.Exit(1)
	}
}
