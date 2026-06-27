// models.go — custom model and provider registry.
//
// Loads ~/.config/tau/agent/models.json (or the path passed to
// LoadModelsFile) and exposes a schema-validated shape that the LLM
// client layer consumes to configure transports, base URLs, and per-model
// overrides.
//
// File format (camelCase keys to match pi):
//
//	{
//	  "providers": {
//	    "ollama": {
//	      "baseUrl": "http://localhost:11434/v1",
//	      "api":     "openai",
//	      "models":  [{"id": "llama3"}]
//	    }
//	  },
//	  "models": [{ "id": "...", "api": "anthropic" }]
//	}

package config

import (
	"errors"
	"fmt"
	"os"
	"strings"
)

// ModelAPI is the protocol family a provider speaks. The runtime currently
// instantiates Anthropic Messages and OpenAI Chat Completions clients;
// the other constants are accepted so users can configure a provider that
// the runtime will reject with a clear error (instead of silently
// misbehaving).
type ModelAPI string

const (
	APIAnthropic ModelAPI = "anthropic"
	APIOpenAI    ModelAPI = "openai"
	APIGemini    ModelAPI = "gemini"
	APIMistral   ModelAPI = "mistral"
	APIBedrock   ModelAPI = "bedrock"
)

// validAPIs is the set of accepted api values.
var validAPIs = map[ModelAPI]bool{
	APIAnthropic: true, APIOpenAI: true, APIGemini: true,
	APIMistral: true, APIBedrock: true,
}

// ModelCost is per-million-token pricing in USD. All four fields are
// required when present; absent fields default to zero.
type ModelCost struct {
	Input      float64 `json:"input,omitempty"`
	Output     float64 `json:"output,omitempty"`
	CacheRead  float64 `json:"cacheRead,omitempty"`
	CacheWrite float64 `json:"cacheWrite,omitempty"`
}

// ModelDefinition is one entry in the registry. Top-level Models slice
// entries need ID + API; provider-attached Models inherit the provider's
// API and may omit it.
type ModelDefinition struct {
	ID               string             `json:"id"`
	Name             string             `json:"name,omitempty"`
	API              ModelAPI           `json:"api,omitempty"`
	BaseURL          string             `json:"baseUrl,omitempty"`
	Reasoning        bool               `json:"reasoning,omitempty"`
	ThinkingLevelMap map[string]*string `json:"thinkingLevelMap,omitempty"`
	Input            []string           `json:"input,omitempty"`
	Cost             *ModelCost         `json:"cost,omitempty"`
	ContextWindow    int                `json:"contextWindow,omitempty"`
	MaxTokens        int                `json:"maxTokens,omitempty"`
	Headers          map[string]string  `json:"headers,omitempty"`
	Compat           map[string]any     `json:"compat,omitempty"`
}

// ProviderDefinition is one entry under providers. Models listed here
// inherit the provider's API/BaseURL/Headers unless overridden.
type ProviderDefinition struct {
	Name           string                     `json:"name,omitempty"`
	BaseURL        string                     `json:"baseUrl,omitempty"`
	APIKey         string                     `json:"apiKey,omitempty"`
	API            ModelAPI                   `json:"api,omitempty"`
	Headers        map[string]string          `json:"headers,omitempty"`
	AuthHeader     bool                       `json:"authHeader,omitempty"`
	Compat         map[string]any             `json:"compat,omitempty"`
	Models         []ModelDefinition          `json:"models,omitempty"`
	ModelOverrides map[string]ModelDefinition `json:"modelOverrides,omitempty"`
}

// ModelsFile is the top-level schema for models.json.
type ModelsFile struct {
	Providers map[string]ProviderDefinition `json:"providers,omitempty"`
	Models    []ModelDefinition             `json:"models,omitempty"`
}

// LoadModelsFile parses path. A missing file returns an empty ModelsFile
// and nil error (first run is normal). Any other read error, malformed
// JSON, or schema violation is returned wrapped in ErrSchemaViolation.
func LoadModelsFile(path string) (*ModelsFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return &ModelsFile{}, nil
		}
		return nil, err
	}
	out := &ModelsFile{}
	if err := strictJSONDecode(data, out); err != nil {
		return nil, err
	}
	if err := out.Validate(); err != nil {
		return nil, err
	}
	return out, nil
}

// Validate checks enum fields and required-id invariants.
func (m *ModelsFile) Validate() error {
	for name, p := range m.Providers {
		if p.API != "" && !validAPIs[p.API] {
			return fmt.Errorf("%w: providers.%s.api %q not in {%s}",
				ErrSchemaViolation, name, p.API, validAPIList())
		}
		for i, md := range p.Models {
			if md.ID == "" {
				return fmt.Errorf("%w: providers.%s.models[%d].id is required",
					ErrSchemaViolation, name, i)
			}
			if md.API != "" && !validAPIs[md.API] {
				return fmt.Errorf("%w: providers.%s.models[%d].api %q not in {%s}",
					ErrSchemaViolation, name, i, md.API, validAPIList())
			}
		}
	}
	for i, md := range m.Models {
		if md.ID == "" {
			return fmt.Errorf("%w: models[%d].id is required",
				ErrSchemaViolation, i)
		}
		if md.API != "" && !validAPIs[md.API] {
			return fmt.Errorf("%w: models[%d].api %q not in {%s}",
				ErrSchemaViolation, i, md.API, validAPIList())
		}
	}
	return nil
}

// KnownModel pairs a ModelDefinition with the provider name that owns it.
// Top-level Models (not attached to a provider) carry an empty Provider.
// Callers that list models for the user (slash commands, TUI pickers) use
// this shape so they can render "provider/id" alongside the model's API.
type KnownModel struct {
	Provider string
	Model    ModelDefinition
}

// AllKnownModels returns a flat list of every model entry in the file,
// combining top-level Models with each provider's Models and
// ModelOverrides. Provider-attached entries inherit the provider's API
// and BaseURL when the model itself leaves them blank, matching the
// resolution semantics of ResolveModel so the listing reflects what the
// runtime would actually wire up.
//
// The slice is ordered: top-level Models first (Provider=""), then each
// provider's entries in map-iteration order. Callers that need stable
// output should sort the result.
func (m *ModelsFile) AllKnownModels() []KnownModel {
	if m == nil {
		return nil
	}
	out := make([]KnownModel, 0, len(m.Models))
	for _, md := range m.Models {
		out = append(out, KnownModel{Provider: "", Model: md})
	}
	for name, p := range m.Providers {
		for _, md := range p.Models {
			merged := mergeModelWithProvider(md, p)
			out = append(out, KnownModel{Provider: name, Model: merged})
		}
		for id, md := range p.ModelOverrides {
			// ModelOverrides is keyed by model id; the entry itself
			// frequently leaves ID empty (the key carries it). Fill in
			// from the map key so the listing is usable for display
			// and for slash-command matching.
			merged := mergeModelWithProvider(md, p)
			if merged.ID == "" {
				merged.ID = id
			}
			out = append(out, KnownModel{Provider: name, Model: merged})
		}
	}
	return out
}

// ResolveModel looks up model id in providers: first under provider-level
// modelOverrides, then under provider-level Models, then under top-level
// Models. Provider fields (BaseURL, Headers, API) are inherited when the
// model itself leaves them empty.
//
// ResolveModel returns nil when id is not found; callers decide whether
// that is an error.
func ResolveModel(providers map[string]ProviderDefinition, topModels []ModelDefinition, provider, id string) *ModelDefinition {
	p, pOk := providers[provider]
	if pOk {
		if override, ok := p.ModelOverrides[id]; ok {
			merged := mergeModelWithProvider(override, p)
			return &merged
		}
		for _, md := range p.Models {
			if md.ID == id {
				merged := mergeModelWithProvider(md, p)
				return &merged
			}
		}
	}
	for _, md := range topModels {
		if md.ID == id {
			out := md // copy
			return &out
		}
	}
	return nil
}

// mergeModelWithProvider fills in missing model fields from the provider.
func mergeModelWithProvider(md ModelDefinition, p ProviderDefinition) ModelDefinition {
	out := md
	if out.API == "" {
		out.API = p.API
	}
	if out.BaseURL == "" {
		out.BaseURL = p.BaseURL
	}
	if out.Headers == nil && len(p.Headers) > 0 {
		out.Headers = make(map[string]string, len(p.Headers))
		for k, v := range p.Headers {
			out.Headers[k] = v
		}
	} else if len(p.Headers) > 0 {
		// Model wins on conflict; provider fills gaps.
		for k, v := range p.Headers {
			if _, ok := out.Headers[k]; !ok {
				out.Headers[k] = v
			}
		}
	}
	return out
}

func validAPIList() string {
	keys := make([]string, 0, len(validAPIs))
	for k := range validAPIs {
		keys = append(keys, string(k))
	}
	return strings.Join(keys, "|")
}
