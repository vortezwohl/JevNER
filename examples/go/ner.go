// JevNER's underlying API, in plain Go — standard library only.
//
// A toy NER loop over one sentence, showing the wire protocol the Python SDK
// is built on:
//   1. Noul question per word   -> "is this word part of an entity?" (probability)
//   2. Choice question per hit  -> "which entity type is it?" (calibrated distribution)
//
// Run:  go run ner.go
// Env:  TYPESAFE_API_KEY (required), TYPESAFE_BASE_URL (default https://api.easyjev.online)
//
// This is a learning example for the protocol. Real NER needs windowing,
// byte-budget safety, caching, taxonomy and nested-tree assembly — that is
// what the `jevner` Python package does.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"regexp"
	"time"
)

const text = "Maria Curie visited Paris and met Doctor Strange."

var (
	baseURL = envOr("TYPESAFE_BASE_URL", "https://api.easyjev.online")
	apiKey  = os.Getenv("TYPESAFE_API_KEY")
	model   = envOr("JEVNER_MODEL", "baseten/mammoth-1")
)

// One typed question; "type" decides the answer shape.
type Question struct {
	Type         string         `json:"type"`
	Instructions map[string]any `json:"instructions"`
	Criteria     map[string]any `json:"criteria"`
}

type Answer struct {
	Type          string             `json:"type"`
	Noul          float64            `json:"noul"`                // present on noul answers
	Choice        string             `json:"choice"`              // present on choice answers
	Confidence    float64            `json:"confidence"`          // present on choice answers
	Probabilities map[string]float64 `json:"probabilities"`       // present on choice answers
}

type systemOneResponse struct {
	Model   string             `json:"model"`
	Answers map[string]Answer  `json:"answers"`
	Usage   map[string]float64 `json:"usage"`
}

// systemOne posts one batch of typed questions against shared state.
func systemOne(state map[string]any, questions map[string]Question) (*systemOneResponse, error) {
	body, err := json.Marshal(map[string]any{"state": state, "model": model, "questions": questions})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost, baseURL+"/v1/systemone", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	// The gateway rejects requests without a User-Agent.
	req.Header.Set("User-Agent", "jevner-example/0.1 (+https://github.com/vortezwohl/JevNER)")

	resp, err := (&http.Client{Timeout: 60 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var raw map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&raw)
		return nil, fmt.Errorf("systemone %d: %v", resp.StatusCode, raw)
	}
	out := new(systemOneResponse)
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return nil, err
	}
	return out, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "set TYPESAFE_API_KEY first")
		os.Exit(1)
	}

	// Whitespace tokens keep this demo dependency-free; the SDK ships a real tokenizer.
	words := regexp.MustCompile(`[^\p{L}\p{N}]`).Split(text, -1)

	// Stage 1: detection — one Noul per word, all in a single request.
	detectQuestions := map[string]Question{}
	for i, word := range words {
		if word == "" {
			continue
		}
		detectQuestions[fmt.Sprintf("w%d", i)] = Question{
			Type:         "noul",
			Instructions: map[string]any{"INPUT": map[string]any{"NAME": word}, "TASK": "Is this word part of a named entity in the text?"},
			Criteria:     map[string]any{"true": "part of a named entity", "false": "not an entity"},
		}
	}
	detect, err := systemOne(map[string]any{"TEXT": text}, detectQuestions)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	type hit struct {
		word string
		noul float64
	}
	var hits []hit
	for i, word := range words {
		if a, ok := detect.Answers[fmt.Sprintf("w%d", i)]; ok && a.Noul >= 0.5 {
			hits = append(hits, hit{word, a.Noul})
		}
	}

	// Stage 2: typing — one Choice per detected word, one more request.
	typingQuestions := map[string]Question{}
	for i, h := range hits {
		typingQuestions[fmt.Sprintf("s%d", i)] = Question{
			Type:         "choice",
			Instructions: map[string]any{"INPUT": map[string]any{"NAME": h.word}, "TASK": "Which entity type fits this span best?"},
			Criteria:     map[string]any{"PERSON": "a named person", "GPE": "a city, state, or country", "NONE": "none of the above"},
		}
	}
	var typing *systemOneResponse
	if len(typingQuestions) > 0 {
		if typing, err = systemOne(map[string]any{"TEXT": text}, typingQuestions); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}

	fmt.Printf("text  : %s\n", text)
	fmt.Printf("model : %s\n", detect.Model)
	for i, h := range hits {
		label := "?"
		if a, ok := typing.Answers[fmt.Sprintf("s%d", i)]; ok {
			label = fmt.Sprintf("%s @ %.2f", a.Choice, math.Round(a.Confidence*100)/100)
		}
		fmt.Printf("%-16s noul=%.2f  %s\n", h.word, h.noul, label)
	}
}
