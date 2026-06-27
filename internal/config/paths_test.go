package config

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// withEnv sets env vars for the duration of fn and restores them after.
func withEnv(t *testing.T, kv map[string]string, fn func()) {
	t.Helper()
	old := make(map[string]string)
	for k, v := range kv {
		old[k] = os.Getenv(k)
		if v == "" {
			os.Unsetenv(k)
		} else {
			os.Setenv(k, v)
		}
	}
	defer func() {
		for k, v := range old {
			if v == "" {
				os.Unsetenv(k)
			} else {
				os.Setenv(k, v)
			}
		}
	}()
	fn()
}

func TestConfigDir_TAU_CONFIG_DIR_Wins(t *testing.T) {
	tmp := t.TempDir()
	withEnv(t, map[string]string{
		"TAU_CONFIG_DIR":  tmp,
		"XDG_CONFIG_HOME": "",
		"APPDATA":         "",
	}, func() {
		got, err := ConfigDir()
		if err != nil {
			t.Fatalf("ConfigDir: %v", err)
		}
		if got != tmp {
			t.Errorf("ConfigDir = %q, want %q", got, tmp)
		}
	})
}

func TestConfigDir_TAU_CONFIG_DIR_EmptyFallsThrough(t *testing.T) {
	// Empty TAU_CONFIG_DIR should be ignored, not returned.
	tmp := t.TempDir()
	withEnv(t, map[string]string{
		"TAU_CONFIG_DIR":  "",
		"XDG_CONFIG_HOME": tmp,
		"APPDATA":         "",
	}, func() {
		got, err := ConfigDir()
		if err != nil {
			t.Fatalf("ConfigDir: %v", err)
		}
		if want := filepath.Join(tmp, "tau"); got != want {
			t.Errorf("ConfigDir = %q, want %q", got, want)
		}
	})
}

func TestConfigDir_XDG_OnPOSIX(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("XDG fallback is POSIX-only")
	}
	tmp := t.TempDir()
	withEnv(t, map[string]string{
		"TAU_CONFIG_DIR":  "",
		"XDG_CONFIG_HOME": tmp,
	}, func() {
		got, err := ConfigDir()
		if err != nil {
			t.Fatalf("ConfigDir: %v", err)
		}
		if want := filepath.Join(tmp, "tau"); got != want {
			t.Errorf("ConfigDir = %q, want %q", got, want)
		}
	})
}

func TestConfigDir_XDG_NotAbsolute_Rejects(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("XDG fallback is POSIX-only")
	}
	withEnv(t, map[string]string{
		"TAU_CONFIG_DIR":  "",
		"XDG_CONFIG_HOME": "relative/path",
	}, func() {
		_, err := ConfigDir()
		if err == nil {
			t.Fatalf("expected error for relative XDG_CONFIG_HOME")
		}
	})
}

func TestConfigDir_AppDataFallback(t *testing.T) {
	// Unset every override so we hit os.UserConfigDir() / os.UserHomeDir().
	// On the test runner, $HOME should be set (CI container), so this should
	// succeed and produce a path ending in /tau.
	withEnv(t, map[string]string{
		"TAU_CONFIG_DIR":  "",
		"XDG_CONFIG_HOME": "",
	}, func() {
		got, err := ConfigDir()
		if err != nil {
			t.Fatalf("ConfigDir: %v", err)
		}
		if !strings.HasSuffix(got, string(os.PathSeparator)+"tau") {
			t.Errorf("ConfigDir = %q, want suffix %q", got, string(os.PathSeparator)+"tau")
		}
	})
}

func TestAgentDir_TAU_AGENT_DIR_Wins(t *testing.T) {
	tmp := t.TempDir()
	withEnv(t, map[string]string{
		"TAU_AGENT_DIR": tmp,
	}, func() {
		got, err := AgentDir()
		if err != nil {
			t.Fatalf("AgentDir: %v", err)
		}
		if got != tmp {
			t.Errorf("AgentDir = %q, want %q", got, tmp)
		}
	})
}

func TestAgentDir_DefaultUnderConfigDir(t *testing.T) {
	tmp := t.TempDir()
	withEnv(t, map[string]string{
		"TAU_CONFIG_DIR": tmp,
		"TAU_AGENT_DIR":  "",
	}, func() {
		got, err := AgentDir()
		if err != nil {
			t.Fatalf("AgentDir: %v", err)
		}
		if want := filepath.Join(tmp, "agent"); got != want {
			t.Errorf("AgentDir = %q, want %q", got, want)
		}
	})
}

func TestEncodeCwd_BasicPOSIX(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX-only path shape")
	}
	cases := []struct {
		in, want string
	}{
		{"/home/user/project", "-home-user-project"},
		{"/a-b/c", "-a-b-c"},
		{"/a/-b-c", "-a--b-c"},
		{"/", "-"},
		{"", "-"},
	}
	for _, c := range cases {
		got := EncodeCwd(c.in)
		if got != c.want {
			t.Errorf("EncodeCwd(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestEncodeCwd_NoCollision(t *testing.T) {
	// Two distinct POSIX paths must never produce the same encoding.
	// (Mixing `/` and `\` in the same string would collide because both
	// separators collapse to `-`, but that never happens within a single
	// OS's path shape, which is all that matters in practice.)
	a := EncodeCwd("/a-b/c")
	b := EncodeCwd("/a/-b-c")
	if a == b {
		t.Errorf("collision: EncodeCwd(/a-b/c)=%q == EncodeCwd(/a/-b-c)=%q", a, b)
	}
	pairs := [][2]string{
		{"/foo", "/foo/"},
		{"/foo/bar", "/foo.bar"},
		{"/foo/bar", "/foo/baz"},
		{"/a", "/a/b"},
	}
	for _, p := range pairs {
		if EncodeCwd(p[0]) == EncodeCwd(p[1]) {
			t.Errorf("collision: %q vs %q both encode to %q", p[0], p[1], EncodeCwd(p[0]))
		}
	}
}

func TestEncodeCwd_AlwaysHasDashPrefix(t *testing.T) {
	for _, in := range []string{"/", "/x", "C:\\Users", "relative"} {
		got := EncodeCwd(in)
		if !strings.HasPrefix(got, "-") {
			t.Errorf("EncodeCwd(%q) = %q, missing dash prefix", in, got)
		}
	}
}

func TestSessionsDir_EncodesCwd(t *testing.T) {
	tmp := t.TempDir()
	withEnv(t, map[string]string{
		"TAU_CONFIG_DIR": tmp,
		"TAU_AGENT_DIR":  "",
	}, func() {
		cwd := "/home/user/project"
		if runtime.GOOS == "windows" {
			cwd = "C:\\Users\\Alice\\project"
		}
		got, err := SessionsDir(cwd)
		if err != nil {
			t.Fatalf("SessionsDir: %v", err)
		}
		// The encoded cwd must appear as the last path segment.
		want := filepath.Join(tmp, "agent", "sessions", EncodeCwd(filepath.Clean(cwd)))
		if got != want {
			t.Errorf("SessionsDir = %q, want %q", got, want)
		}
	})
}

func TestSessionsDir_EmptyCwd_Errors(t *testing.T) {
	if _, err := SessionsDir(""); err == nil {
		t.Errorf("SessionsDir(\"\") should error")
	}
}

func TestPluginsDir_ReturnsGlobalAndProject(t *testing.T) {
	tmp := t.TempDir()
	withEnv(t, map[string]string{"TAU_CONFIG_DIR": tmp}, func() {
		got, err := PluginsDir("/some/cwd")
		if err != nil {
			t.Fatalf("PluginsDir: %v", err)
		}
		if len(got) != 2 {
			t.Fatalf("PluginsDir returned %d entries, want 2", len(got))
		}
		wantGlobal := filepath.Join(tmp, "plugins")
		if got[0] != wantGlobal {
			t.Errorf("PluginsDir[0] = %q, want %q", got[0], wantGlobal)
		}
		// The project entry must be under the cwd's .tau/plugins.
		if !strings.HasSuffix(got[1], filepath.Join(".tau", "plugins")) {
			t.Errorf("PluginsDir[1] = %q, want suffix .tau/plugins", got[1])
		}
	})
}

func TestPluginsDir_EmptyCwd_OnlyGlobal(t *testing.T) {
	tmp := t.TempDir()
	withEnv(t, map[string]string{"TAU_CONFIG_DIR": tmp}, func() {
		got, err := PluginsDir("")
		if err != nil {
			t.Fatalf("PluginsDir: %v", err)
		}
		if len(got) != 1 {
			t.Fatalf("PluginsDir(\"\") returned %d entries, want 1", len(got))
		}
	})
}

func TestMkdirAll_CreatesWith0700(t *testing.T) {
	tmp := t.TempDir()
	target := filepath.Join(tmp, "a", "b", "c")
	if err := MkdirAll(target); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	info, err := os.Stat(target)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	// AND of group/other bits should be zero. (We don't check exact mode
	// because the umask may strip bits we wrote — but we explicitly pass
	// 0o700, which has no group/other bits to begin with.)
	if info.Mode()&0o077 != 0 {
		t.Errorf("mode = %v, want no group/other bits", info.Mode())
	}
}
