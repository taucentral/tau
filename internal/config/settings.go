// settings.go — Settings schema for tau.
//
// The Settings struct mirrors pi's settings shape (see pi's
// settings-manager.ts:80) so users can carry muscle memory across both
// tools. JSON tags are camelCase to match pi's on-disk format.
//
// Sub-structs are kept in this file in the same order they appear in pi
// so cross-reference is mechanical.

package config

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ThinkingLevel is the runtime thinking-effort selector.
type ThinkingLevel string

const (
	ThinkingOff     ThinkingLevel = "off"
	ThinkingMinimal ThinkingLevel = "minimal"
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXHigh   ThinkingLevel = "xhigh"
)

// validThinkingLevels is the set of allowed ThinkingLevel values.
var validThinkingLevels = map[ThinkingLevel]bool{
	ThinkingOff: true, ThinkingMinimal: true, ThinkingLow: true,
	ThinkingMedium: true, ThinkingHigh: true, ThinkingXHigh: true,
}

// TransportSetting selects the streaming transport for provider streams.
type TransportSetting string

const (
	TransportAuto      TransportSetting = "auto"
	TransportSSE       TransportSetting = "sse"
	TransportWebsocket TransportSetting = "websocket"
)

// validTransports is the set of allowed TransportSetting values.
var validTransports = map[TransportSetting]bool{
	TransportAuto: true, TransportSSE: true, TransportWebsocket: true,
}

// SteeringMode selects how multiple steering agents are dispatched.
type SteeringMode string

const (
	SteeringAll        SteeringMode = "all"
	SteeringOneAtATime SteeringMode = "one-at-a-time"
)

// validSteeringModes is the set of allowed SteeringMode values.
var validSteeringModes = map[SteeringMode]bool{
	SteeringAll: true, SteeringOneAtATime: true,
}

// CompactionSettings controls the multi-stage compaction pipeline.
type CompactionSettings struct {
	Enabled          *bool `json:"enabled,omitempty"`
	ReserveTokens    *int  `json:"reserveTokens,omitempty"`
	KeepRecentTokens *int  `json:"keepRecentTokens,omitempty"`
}

// ToolsSettings configures lazy-tool hydration. All fields are optional;
// nil means "use the zero-value default" which preserves eager rendering
// for tools that do not implement LazyHeadlessTool.
//
// Fields:
//   - HydrationMode (default "heuristic"): how the registry evaluates
//     LazyHeadlessTool triggers. One of "heuristic", "model_declared",
//     "off". "off" disables lazy registration entirely.
//   - AlwaysRender: tool names that always render regardless of the
//     heuristic. Per-deployment escape hatch for tools the workflow
//     depends on.
//   - RecentUseWindow (default 5): how many turns of tool-call history
//     count as "recent" for the recency trigger.
type ToolsSettings struct {
	HydrationMode   *string  `json:"hydrationMode,omitempty"`
	AlwaysRender    []string `json:"alwaysRender,omitempty"`
	RecentUseWindow *int     `json:"recentUseWindow,omitempty"`
}

// PromptsSettings configures the system-prompt Assembler's ancestor
// walk. All fields are optional; nil means "use the zero-value default"
// which preserves the pre-walk behavior for projects whose cwd is the
// repo root.
//
// Fields:
//   - WalkToRoot (default false): when true, ignore VCS markers and
//     walk all the way to the filesystem root.
//   - MaxAncestorDepth (default 0 = unlimited): cap on ancestor
//     directories visited. 1 reads only cwd and cwd/...
//   - StopDir (default ""): explicit stop directory. When set,
//     overrides both the marker scan and WalkToRoot.
type PromptsSettings struct {
	WalkToRoot       *bool   `json:"walkToRoot,omitempty"`
	MaxAncestorDepth *int    `json:"maxAncestorDepth,omitempty"`
	StopDir          *string `json:"stopDir,omitempty"`
}

// BranchSummarySettings controls the branch-summary prompt.
type BranchSummarySettings struct {
	ReserveTokens *int  `json:"reserveTokens,omitempty"`
	SkipPrompt    *bool `json:"skipPrompt,omitempty"`
}

// ProviderRetrySettings is the per-provider retry/timeout policy.
type ProviderRetrySettings struct {
	TimeoutMs       *int `json:"timeoutMs,omitempty"`
	MaxRetries      *int `json:"maxRetries,omitempty"`
	MaxRetryDelayMs *int `json:"maxRetryDelayMs,omitempty"`
}

// RetrySettings is the top-level retry policy.
type RetrySettings struct {
	Enabled     *bool                  `json:"enabled,omitempty"`
	MaxRetries  *int                   `json:"maxRetries,omitempty"`
	BaseDelayMs *int                   `json:"baseDelayMs,omitempty"`
	Provider    *ProviderRetrySettings `json:"provider,omitempty"`
}

// TerminalSettings configures terminal rendering behavior.
type TerminalSettings struct {
	ShowImages           *bool `json:"showImages,omitempty"`
	ImageWidthCells      *int  `json:"imageWidthCells,omitempty"`
	ClearOnShrink        *bool `json:"clearOnShrink,omitempty"`
	ShowTerminalProgress *bool `json:"showTerminalProgress,omitempty"`
}

// ImageSettings configures image-handling policy.
type ImageSettings struct {
	AutoResize  *bool `json:"autoResize,omitempty"`
	BlockImages *bool `json:"blockImages,omitempty"`
}

// ThinkingBudgetsSettings overrides per-level thinking budgets.
type ThinkingBudgetsSettings struct {
	Minimal *int `json:"minimal,omitempty"`
	Low     *int `json:"low,omitempty"`
	Medium  *int `json:"medium,omitempty"`
	High    *int `json:"high,omitempty"`
}

// MarkdownSettings configures markdown rendering details.
type MarkdownSettings struct {
	CodeBlockIndent *string `json:"codeBlockIndent,omitempty"`
}

// WarningSettings toggles non-fatal warnings.
type WarningSettings struct {
	AnthropicExtraUsage *bool `json:"anthropicExtraUsage,omitempty"`
}

// Settings is the full settings tree loaded from settings.json. Fields are
// pointers so we can distinguish "unset" (use parent default) from "set
// to zero" (override the default). Pointer types also let the deep-merge
// implementation in settings_storage.go distinguish "leave alone" from
// "replace with zero".
type Settings struct {
	LastChangelogVersion      *string                  `json:"lastChangelogVersion,omitempty"`
	DefaultProvider           *string                  `json:"defaultProvider,omitempty"`
	DefaultModel              *string                  `json:"defaultModel,omitempty"`
	DefaultThinkingLevel      *ThinkingLevel           `json:"defaultThinkingLevel,omitempty"`
	Transport                 *TransportSetting        `json:"transport,omitempty"`
	SteeringMode              *SteeringMode            `json:"steeringMode,omitempty"`
	FollowUpMode              *SteeringMode            `json:"followUpMode,omitempty"`
	Theme                     *string                  `json:"theme,omitempty"`
	Compaction                *CompactionSettings      `json:"compaction,omitempty"`
	Tools                     *ToolsSettings           `json:"tools,omitempty"`
	BranchSummary             *BranchSummarySettings   `json:"branchSummary,omitempty"`
	Retry                     *RetrySettings           `json:"retry,omitempty"`
	HideThinkingBlock         *bool                    `json:"hideThinkingBlock,omitempty"`
	ShellPath                 *string                  `json:"shellPath,omitempty"`
	QuietStartup              *bool                    `json:"quietStartup,omitempty"`
	DefaultProjectTrust       *string                  `json:"defaultProjectTrust,omitempty"`
	ShellCommandPrefix        *string                  `json:"shellCommandPrefix,omitempty"`
	NpmCommand                []string                 `json:"npmCommand,omitempty"`
	CollapseChangelog         *bool                    `json:"collapseChangelog,omitempty"`
	EnableInstallTelemetry    *bool                    `json:"enableInstallTelemetry,omitempty"`
	EnableAnalytics           *bool                    `json:"enableAnalytics,omitempty"`
	TrackingID                *string                  `json:"trackingId,omitempty"`
	Packages                  []any                    `json:"packages,omitempty"`
	Extensions                []string                 `json:"extensions,omitempty"`
	Skills                    []string                 `json:"skills,omitempty"`
	Prompts                   *PromptsSettings         `json:"prompts,omitempty"`
	Themes                    []string                 `json:"themes,omitempty"`
	EnableSkillCommands       *bool                    `json:"enableSkillCommands,omitempty"`
	Terminal                  *TerminalSettings        `json:"terminal,omitempty"`
	Images                    *ImageSettings           `json:"images,omitempty"`
	EnabledModels             []string                 `json:"enabledModels,omitempty"`
	DoubleEscapeAction        *string                  `json:"doubleEscapeAction,omitempty"`
	TreeFilterMode            *string                  `json:"treeFilterMode,omitempty"`
	ThinkingBudgets           *ThinkingBudgetsSettings `json:"thinkingBudgets,omitempty"`
	EditorPaddingX            *int                     `json:"editorPaddingX,omitempty"`
	AutocompleteMaxVisible    *int                     `json:"autocompleteMaxVisible,omitempty"`
	ShowHardwareCursor        *bool                    `json:"showHardwareCursor,omitempty"`
	Markdown                  *MarkdownSettings        `json:"markdown,omitempty"`
	Warnings                  *WarningSettings         `json:"warnings,omitempty"`
	SessionDir                *string                  `json:"sessionDir,omitempty"`
	HTTPProxy                 *string                  `json:"httpProxy,omitempty"`
	HTTPIdleTimeoutMs         *int                     `json:"httpIdleTimeoutMs,omitempty"`
	WebSocketConnectTimeoutMs *int                     `json:"websocketConnectTimeoutMs,omitempty"`
	Keybindings               map[string]Keybinding    `json:"keybindings,omitempty"`
}

// DefaultSettings returns the canonical defaults applied when no override
// exists. Returned values are fresh copies; mutating them does not affect
// future calls.
func DefaultSettings() Settings {
	enabled := true
	off := ThinkingOff
	x := 8192
	keep := 20000
	transport := TransportAuto
	steering := SteeringOneAtATime
	maxRetries := 4
	providerMaxDelay := 30000
	providerMaxRetries := 4
	baseDelayMs := 2000
	showImages := true
	imageWidth := 60
	autoResize := true
	enableSkillCommands := true
	enableInstallTelemetry := true
	anthropicExtraUsage := true
	editorPad := 0
	autocomplete := 5
	reserveBranch := 16384
	skipBranchPrompt := false
	codeBlockIndent := "  "
	defaultTrust := "ask"
	toolsHydrationMode := "heuristic"
	toolsRecentUseWindow := 5

	return Settings{
		Transport:            &transport,
		SteeringMode:         &steering,
		FollowUpMode:         &steering,
		DefaultThinkingLevel: &off,
		DefaultProjectTrust:  &defaultTrust,
		Compaction: &CompactionSettings{
			Enabled:          &enabled,
			ReserveTokens:    &x,
			KeepRecentTokens: &keep,
		},
		Tools: &ToolsSettings{
			HydrationMode:   &toolsHydrationMode,
			RecentUseWindow: &toolsRecentUseWindow,
		},
		BranchSummary: &BranchSummarySettings{
			ReserveTokens: &reserveBranch,
			SkipPrompt:    &skipBranchPrompt,
		},
		Retry: &RetrySettings{
			Enabled:     &enabled,
			MaxRetries:  &maxRetries,
			BaseDelayMs: &baseDelayMs,
			Provider: &ProviderRetrySettings{
				MaxRetries:      &providerMaxRetries,
				MaxRetryDelayMs: &providerMaxDelay,
			},
		},
		Terminal: &TerminalSettings{
			ShowImages:      &showImages,
			ImageWidthCells: &imageWidth,
		},
		Images: &ImageSettings{
			AutoResize: &autoResize,
		},
		Markdown: &MarkdownSettings{
			CodeBlockIndent: &codeBlockIndent,
		},
		Warnings: &WarningSettings{
			AnthropicExtraUsage: &anthropicExtraUsage,
		},
		EnableSkillCommands:    &enableSkillCommands,
		EnableInstallTelemetry: &enableInstallTelemetry,
		EditorPaddingX:         &editorPad,
		AutocompleteMaxVisible: &autocomplete,
	}
}

// ValidateSettings checks enum fields and reports the first violation as
// a ErrSchemaViolation wrapper, or nil if everything is well-formed.
func (s Settings) Validate() error {
	if s.DefaultThinkingLevel != nil {
		if !validThinkingLevels[*s.DefaultThinkingLevel] {
			return fmt.Errorf("%w: defaultThinkingLevel %q not in {%s}",
				ErrSchemaViolation, *s.DefaultThinkingLevel,
				"off|minimal|low|medium|high|xhigh")
		}
	}
	if s.Transport != nil {
		if !validTransports[*s.Transport] {
			return fmt.Errorf("%w: transport %q not in {%s}",
				ErrSchemaViolation, *s.Transport,
				"auto|sse|websocket")
		}
	}
	if s.SteeringMode != nil {
		if !validSteeringModes[*s.SteeringMode] {
			return fmt.Errorf("%w: steeringMode %q not in {%s}",
				ErrSchemaViolation, *s.SteeringMode,
				"all|one-at-a-time")
		}
	}
	if s.FollowUpMode != nil {
		if !validSteeringModes[*s.FollowUpMode] {
			return fmt.Errorf("%w: followUpMode %q not in {%s}",
				ErrSchemaViolation, *s.FollowUpMode,
				"all|one-at-a-time")
		}
	}
	if s.DefaultProjectTrust != nil {
		switch *s.DefaultProjectTrust {
		case "ask", "always", "never":
		default:
			return fmt.Errorf("%w: defaultProjectTrust %q not in {ask,always,never}",
				ErrSchemaViolation, *s.DefaultProjectTrust)
		}
	}
	return nil
}

// strictJSONDecode decodes data into v using DisallowUnknownFields so
// typos in settings become loud errors instead of being silently ignored.
func strictJSONDecode(data []byte, v any) error {
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("%w: %v", ErrSchemaViolation, err)
	}
	return nil
}
